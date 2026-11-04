"""factorlab 单元测试 (合成数据, 不依赖课程原始行情).

运行: py -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from factorlab.data import Panel, half_up, limit_pct                    # noqa: E402
from factorlab import evaluate as E                                      # noqa: E402
from factorlab import factors as F                                       # noqa: E402
from factorlab.engine import Backtest, Book                              # noqa: E402


def make_panel(n_days=80, n_codes=8, seed=0, st_codes=()):
    """写 7 个 feather 到临时目录并读成 Panel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-02', periods=n_days)
    codes = [f'{600000 + i}.SH' for i in range(n_codes)]
    price = 10.0 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_codes)), axis=0)
    close = pd.DataFrame(price, index=dates, columns=codes)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    amt = pd.DataFrame(rng.uniform(1e8, 2e8, close.shape), index=dates, columns=codes)
    ntr = pd.DataFrame(rng.integers(1000, 5000, close.shape).astype(float),
                       index=dates, columns=codes)
    cumadj = pd.DataFrame(1.0, index=dates, columns=codes)
    tmp = tempfile.mkdtemp()
    for name, df in [('open', open_), ('high', high), ('low', low), ('close', close),
                     ('amt', amt), ('num_trades', ntr), ('cumadj', cumadj)]:
        df.reset_index().rename(columns={'index': 'date'}).to_feather(f'{tmp}/{name}.feather')
    st = pd.DataFrame(0, index=dates, columns=codes, dtype='int8')
    for c in st_codes:
        st[c] = 1
    st.to_parquet(f'{tmp}/st.parquet')
    return Panel(tmp, f'{tmp}/st.parquet', None), close, dates, codes


class TestData(unittest.TestCase):
    def test_half_up_matches_exchange(self):
        # 昨收 5.75 的涨停价, 交易所 6.33, 银行家舍入会得 6.32
        self.assertAlmostEqual(float(half_up(5.75 * 1.1)), 6.33)
        self.assertAlmostEqual(float(half_up(10.005)), 10.01)
        self.assertAlmostEqual(float(half_up(1.004)), 1.00)

    def test_limit_pct_by_board(self):
        r = limit_pct(['600000.SH', '300001.SZ', '688001.SH', '000001.SZ'])
        self.assertEqual(list(r), [0.10, 0.20, 0.20, 0.10])

    def test_panel_alignment(self):
        p, close, dates, codes = make_panel()
        self.assertEqual(p.close.shape, close.shape)
        self.assertTrue(np.allclose(p.pstar, p.close * p.cumadj))

    def test_st_unknown_excluded(self):
        p, _, _, _ = make_panel(st_codes=('600000.SH',))
        t = 60
        u = p.universe(t, min_hist=10, min_amt20=0)
        idx = np.nonzero(u)[0]
        self.assertNotIn(p.codes.index('600000.SH'), idx)
        self.assertEqual(len(idx), len(p.codes) - 1)


class TestLabels(unittest.TestCase):
    def test_fwd_label_alignment(self):
        p, close, dates, codes = make_panel(n_days=20, n_codes=1)
        lab = E.fwd_labels(p, 2)
        t = 5
        want = p.opstar[t + 3, 0] / p.opstar[t + 1, 0] - 1
        self.assertAlmostEqual(float(lab[t, 0]), float(want), places=6)
        # 尾部必须有 NaN, 不能拿未来数据
        self.assertTrue(np.isnan(lab[-1, 0]))
        self.assertTrue(np.isnan(lab[-2, 0]))

    def test_ic_perfect_and_reversed(self):
        rng = np.random.default_rng(1)
        lab = rng.normal(size=(5, 300))
        fac = lab.copy()
        mask = np.ones_like(lab, dtype=bool)
        st = E.factor_stats(fac, lab, mask, np.arange(5), 1)
        self.assertAlmostEqual(st['rank_ic_mean'], 1.0, places=6)
        st2 = E.factor_stats(-fac, lab, mask, np.arange(5), 1)
        self.assertAlmostEqual(st2['rank_ic_mean'], -1.0, places=6)

    def test_quantile_monotone(self):
        rng = np.random.default_rng(2)
        lab = rng.normal(size=(30, 500))
        fac = lab + rng.normal(0, 0.01, lab.shape)
        mask = np.ones_like(lab, dtype=bool)
        st = E.factor_stats(fac, lab, mask, np.arange(30), 1)
        q = st['quantiles']
        self.assertTrue(all(q[i] < q[i + 1] for i in range(4)))
        self.assertGreater(st['monotonic'], 0.9)


class TestStats(unittest.TestCase):
    def test_nw_t_sane(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 2000)
        t, p = E.nw_t(x, lag=5)
        self.assertLess(abs(t), 4.0)
        self.assertGreater(p, 0.001)
        y = rng.normal(0.2, 1, 2000)
        t2, p2 = E.nw_t(y, lag=5)
        self.assertGreater(t2, 5.0)
        self.assertLess(p2, 1e-6)

    def test_bh_fdr(self):
        p = np.array([0.001, 0.01, 0.02, 0.5, np.nan])
        q = E.bh_fdr(p)
        self.assertTrue(np.isnan(q[-1]))
        self.assertTrue(np.all(np.diff(q[:4][np.argsort(p[:4])]) >= -1e-12))
        self.assertLessEqual(q[0], 0.001 * 4)


class TestEngine(unittest.TestCase):
    def _book(self):
        p, close, dates, codes = make_panel(n_days=40, n_codes=5)
        b = Book(p, init_cash=1e6, commission=1e-4, min_commission=5.0,
                 stamp=5e-4, slippage=0.0, participation=1.0)
        return p, b, dates

    def test_buy_fee_and_cash(self):
        p, b, _ = self._book()
        i, sig, j = 25, 24, 0
        cash0 = b.cash
        gross = b.buy(i, sig, j, 1e5)
        self.assertGreater(gross, 0)
        tr = b.trades[-1]
        self.assertAlmostEqual(tr[6], max(gross * 1e-4, 5.0), places=6)
        self.assertAlmostEqual(tr[7], 0.0)                       # 买入无印花税
        self.assertAlmostEqual(b.cash, cash0 - gross - tr[6], places=6)
        self.assertEqual(int(tr[3]) % 100, 0)                    # 整手

    def test_sell_fee_and_stamp(self):
        p, b, _ = self._book()
        i, sig, j = 25, 24, 0
        b.buy(i, sig, j, 1e5)
        cash1 = b.cash
        held = b.units[j] * p.cumadj[i + 1, j]
        b.sell(i + 1, i, j, held * p.open[i + 1, j], full=True)
        tr = b.trades[-1]
        self.assertAlmostEqual(tr[6], max(tr[5] * 1e-4, 5.0), places=6)
        self.assertAlmostEqual(tr[7], tr[5] * 5e-4, places=6)
        self.assertAlmostEqual(b.cash, cash1 + tr[5] - tr[6] - tr[7], places=6)
        self.assertAlmostEqual(b.units[j], 0.0, places=12)

    def test_t_plus_1(self):
        p, b, _ = self._book()
        i, sig, j = 25, 24, 0
        b.buy(i, sig, j, 1e5)
        held = b.units[j] * p.cumadj[i, j]
        b.sell(i, sig, j, held * p.open[i, j], full=True)        # 同日卖出应被拒
        self.assertEqual(b.trades[-1][2], 'BUY')
        self.assertTrue(any(r[4] == 't_plus_1' for r in b.rejects))

    def test_limit_up_blocks_buy(self):
        p, b, _ = self._book()
        i, sig, j = 25, 24, 0
        p.open[i, j] = p.limit_up[i, j] * 1.0001
        p.open[i, j] = p.limit_up[i, j]
        self.assertEqual(b.buy(i, sig, j, 1e5), 0.0)
        self.assertTrue(any(r[4] == 'limit_up' for r in b.rejects))

    def test_participation_cap(self):
        p, b, _ = self._book()
        b.participation = 0.01
        i, sig, j = 25, 24, 0
        gross = b.buy(i, sig, j, 1e7)
        cap = 0.01 * b._amt20[sig, j]
        self.assertLessEqual(gross, cap + 1e-6)

    def test_cash_never_negative(self):
        p, b, _ = self._book()
        i, sig, j = 25, 24, 0
        b.buy(i, sig, j, 1e12)
        self.assertGreaterEqual(b.cash, -1e-6)


class TestBacktest(unittest.TestCase):
    def test_lot_sell_and_liquidation(self):
        p, close, dates, codes = make_panel(n_days=60, n_codes=6)
        rng = np.random.default_rng(7)
        fac = rng.normal(size=p.close.shape).astype(np.float32)

        class Long:
            def pool(self, pp, sig):
                return pp.universe(sig, min_hist=5, min_amt20=0) & np.isfinite(fac[sig])

            def target_weights(self, pp, sig):
                idx = np.nonzero(self.pool(pp, sig))[0]
                if len(idx) == 0:
                    return {}
                s = fac[sig, idx]
                pick = idx[np.argsort(-s)[:2]]
                return {int(j): 0.5 for j in pick}

        bt = Backtest(p, Long(), start='2020-02-03', end='2020-03-31', rebal=5,
                      exit_days=2, init_cash=1e6, min_commission=5.0)
        eq, tr, rj = bt.run()
        self.assertGreater(len(tr), 4)
        self.assertAlmostEqual(float(eq['nav'].iloc[-1]),
                               float(eq['equity'].iloc[-1] / 1e6), places=9)
        # 最后一天不允许再有持仓
        self.assertEqual(int(eq['n_hold'].iloc[-1]), 0)
        # 卖出股数不超过持仓
        self.assertTrue((tr[tr.side == 'SELL']['shares'] > 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
