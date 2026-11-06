"""evolve 模块单元测试 (合成数据).

运行: py -m unittest discover -s tests -v
"""
import os
import random
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from factorlab.data import Panel                            # noqa: E402
from factorlab.evolve import expr                           # noqa: E402
from factorlab.evolve.mutate import mutate, crossover, random_tree, replace_at  # noqa: E402
from factorlab.evolve.score import score_candidate, sampled_ranks   # noqa: E402


def make_panel(n_days=90, n_codes=260, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-02', periods=n_days)
    codes = [f'{600000 + i}.SH' for i in range(n_codes)]
    price = 10.0 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_codes)), axis=0)
    close = pd.DataFrame(price, index=dates, columns=codes)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    amt = pd.DataFrame(rng.uniform(1e8, 3e8, close.shape), index=dates, columns=codes)
    ntr = pd.DataFrame(rng.integers(500, 5000, close.shape).astype(float),
                       index=dates, columns=codes)
    cumadj = pd.DataFrame(1.0, index=dates, columns=codes)
    tmp = tempfile.mkdtemp()
    for name, df in [('open', open_), ('high', high), ('low', low), ('close', close),
                     ('amt', amt), ('num_trades', ntr), ('cumadj', cumadj)]:
        df.reset_index().rename(columns={'index': 'date'}).to_feather(f'{tmp}/{name}.feather')
    st = pd.DataFrame(0, index=dates, columns=codes, dtype='int8')
    st.to_parquet(f'{tmp}/st.parquet')
    return Panel(tmp, f'{tmp}/st.parquet', None)


class TestExpr(unittest.TestCase):
    def setUp(self):
        self.p = make_panel()
        self.ctx = expr.Ctx(self.p)

    def test_nodes_and_str(self):
        n = ('u', 'neg', ('t', 'pct', ('v', 'pstar'), 5))
        self.assertEqual(expr.nodes(n), 3)
        self.assertEqual(expr.depth(n), 3)
        self.assertEqual(expr.to_str(n), 'neg(pct(pstar, 5))')
        self.assertEqual(expr.windows(n), [5])

    def test_eval_matches_numpy(self):
        n = ('u', 'neg', ('t', 'pct', ('v', 'pstar'), 5))
        out = self.ctx.evaluate(n)
        want = -(self.p.pstar[5:] / self.p.pstar[:-5] - 1.0)
        self.assertTrue(np.allclose(out[5:], want, atol=1e-5, equal_nan=True))

    def test_constant_broadcasts(self):
        out = self.ctx.evaluate(('c', 3.0))
        self.assertEqual(out.shape, self.p.pstar.shape)
        self.assertTrue(np.all(out == 3.0))

    def test_no_lookahead(self):
        """把未来数据改掉, 过去的因子值不能变."""
        n = ('t', 'mean', ('v', 'ret'), 10)
        a = self.ctx.evaluate(n).copy()
        p2 = make_panel()
        p2.close[60:] *= 1.5
        p2.pstar = p2.close * p2.cumadj
        p2.ret[1:] = p2.pstar[1:] / p2.pstar[:-1] - 1.0
        ctx2 = expr.Ctx(p2)
        b = ctx2.evaluate(n)
        self.assertTrue(np.allclose(a[:55], b[:55], atol=1e-6, equal_nan=True))

    def test_sanitize_inf(self):
        x = np.array([[1.0, np.inf, -np.inf, np.nan]], dtype=np.float32)
        y = expr.sanitize(x)
        self.assertTrue(np.isnan(y[0, 1]) and np.isnan(y[0, 2]))


class TestMutate(unittest.TestCase):
    def test_random_tree_size(self):
        rng = random.Random(1)
        for _ in range(30):
            t = random_tree(rng, 6)
            self.assertLessEqual(expr.nodes(t), 10)
            self.assertGreaterEqual(expr.nodes(t), 1)

    def test_mutate_keeps_valid(self):
        rng = random.Random(2)
        n = ('u', 'neg', ('t', 'std', ('v', 'ret'), 20))
        for _ in range(50):
            m = mutate(rng, n, 10)
            expr.to_str(m)          # 不抛异常即为结构合法
            self.assertLessEqual(expr.nodes(m), 14)

    def test_crossover(self):
        rng = random.Random(3)
        a = ('t', 'mean', ('v', 'amt'), 20)
        b = ('u', 'neg', ('t', 'std', ('v', 'ret'), 5))
        c = crossover(rng, a, b)
        expr.to_str(c)

    def test_replace_at_roundtrip(self):
        n = ('b', 'div', ('v', 'close'), ('t', 'mean', ('v', 'amt'), 20))
        new = ('v', 'open')
        m = replace_at(n, (1, 0), new)
        self.assertEqual(expr.to_str(m), '(close / mean(open, 20))')


class TestScore(unittest.TestCase):
    def test_perfect_factor(self):
        p = make_panel()
        ctx = expr.Ctx(p)
        from factorlab import evaluate as E
        label = E.fwd_labels(p, 1)
        days = np.arange(20, 80)
        masks = np.zeros(p.close.shape, dtype=bool)
        for t in days:
            masks[t] = True
        fac = label.copy()                     # 完全预测
        st = score_candidate(fac, label, masks, days, 1)
        self.assertIsNotNone(st)
        self.assertGreater(st['rank_ic'], 0.99)

    def test_sampled_ranks(self):
        p = make_panel()
        ctx = expr.Ctx(p)
        fac = ctx.evaluate(('t', 'pct', ('v', 'pstar'), 5))
        masks = np.ones(p.close.shape, dtype=bool)
        R = sampled_ranks(fac, masks, np.arange(20, 40))
        self.assertEqual(R.shape, (20, len(p.codes)))
        self.assertTrue(np.nanmax(R) <= len(p.codes))


if __name__ == '__main__':
    unittest.main(verbosity=2)
