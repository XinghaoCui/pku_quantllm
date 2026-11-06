"""演化 harness: 固定预算下的四组对照.

  基线   baseline   只保留一个"当前最优", 子代均匀变异, |RankIC| 更高就替换
  harness harness   因子池 + 多样性 + 相关/增量门控 + 按接受率调整变异算子
  学习   learning    在 harness 上再加一个候选评分器: 先用结构特征预测质量,
                    只把昂贵的完整评价花在预测更好的候选上
  随机   random      纯随机公式, 不迭代 —— 用来估计"搜索次数本身"能造出多好的假象

四组总完整评价次数相同 (budget), 使用同一训练区间、同一标签、同一股票池.
"""
from __future__ import annotations

import random

import numpy as np

from . import expr
from .mutate import crossover, mutate, random_tree
from .score import incremental_ic, max_corr, sampled_ranks, score_candidate

SEEDS = [
    ('rev5',         ('u', 'neg', ('t', 'pct', ('v', 'pstar'), 5))),
    ('lowvol20',     ('u', 'neg', ('t', 'std', ('v', 'ret'), 20))),
    ('lowamt20',     ('u', 'neg', ('u', 'log', ('t', 'mean', ('v', 'amt'), 20)))),
    ('mom60',        ('t', 'pct', ('v', 'pstar'), 60)),
    ('lowntr20',     ('u', 'neg', ('u', 'log', ('t', 'mean', ('v', 'ntr'), 20)))),
    ('intraday_rev', ('u', 'neg', ('t', 'mean', ('v', 'intraday'), 5))),
    ('amt_shrink',   ('b', 'sub', ('c', 1.0),
                      ('b', 'div', ('t', 'mean', ('v', 'amt'), 5),
                       ('t', 'mean', ('v', 'amt'), 20)))),
    ('range_low20',  ('u', 'neg', ('b', 'div',
                       ('b', 'sub', ('v', 'pstar'), ('t', 'min', ('v', 'pstar'), 20)),
                       ('b', 'add', ('c', 1e-6),
                        ('b', 'sub', ('t', 'max', ('v', 'pstar'), 20),
                         ('t', 'min', ('v', 'pstar'), 20)))))),
]


class Candidate:
    __slots__ = ('cid', 'arm', 'rnd', 'node', 'parent', 'stats', 'accepted',
                 'reason', 'corr', 'incr', 'rank', 'pred')

    def __init__(self, cid, arm, rnd, node, parent=None):
        self.cid = cid
        self.arm = arm
        self.rnd = rnd
        self.node = node
        self.parent = parent
        self.stats = None
        self.accepted = False
        self.reason = ''
        self.corr = np.nan
        self.incr = np.nan
        self.rank = None
        self.pred = np.nan

    @property
    def ic(self):
        return abs(self.stats['rank_ic']) if self.stats else 0.0

    def row(self):
        s = self.stats or {}
        return {
            'cid': self.cid, 'arm': self.arm, 'round': self.rnd,
            'parent': self.parent or '', 'formula': expr.to_str(self.node),
            'nodes': expr.nodes(self.node), 'depth': expr.depth(self.node),
            'rank_ic': s.get('rank_ic'), 't': s.get('t'), 'icir': s.get('icir'),
            'pos': s.get('pos'), 'mono': s.get('mono'), 'ls': s.get('ls'),
            'n_days': s.get('n_days'), 'avg_names': s.get('avg_names'),
            'max_corr': None if not np.isfinite(self.corr) else round(float(self.corr), 4),
            'incr_ic': None if not np.isfinite(self.incr) else round(float(self.incr), 4),
            'accepted': self.accepted, 'reason': self.reason,
            'direction': 1.0 if (s.get('rank_ic') or 0) >= 0 else -1.0,
            'pred': None if not np.isfinite(self.pred) else round(float(self.pred), 4),
        }


class Evolution:
    def __init__(self, ctx, train_days, sample_days, label, masks,
                 budget=64, rounds=8, seed=2026, pool_size=8):
        self.ctx = ctx
        self.days = np.asarray(train_days)
        self.sample_days = np.asarray(sample_days)
        self.label = label
        self.masks = masks
        self.budget = budget
        self.rounds = rounds
        self.pool_size = pool_size
        self.seed = seed
        self.rng = random.Random(seed)
        self._n = 0
        self.cands = []
        self.learn_X, self.learn_y = [], []

    # -------------------------------------------------------------- 评价
    def _cid(self):
        self._n += 1
        return f'c{self._n:04d}'

    def evaluate(self, node, arm, rnd, parent=None, with_rank=True):
        c = Candidate(self._cid(), arm, rnd, node, parent)
        fac = expr.sanitize(self.ctx.evaluate(node))
        st = score_candidate(fac, self.label, self.masks, self.days)
        if st is None:
            c.reason = 'no_stats'
            self.cands.append(c)
            return c
        c.stats = st
        if with_rank:
            c.rank = sampled_ranks(fac, self.masks, self.sample_days)
        self.cands.append(c)
        return c

    def _composite(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nanmean(np.stack([m.rank for m in self.pool]), axis=0)

    # -------------------------------------------------------------- 门控
    def gate(self, c, composite, strict=True):
        st = c.stats
        if st is None:
            return False, 'no_stats'
        if st['avg_names'] < 800:
            return False, 'coverage'
        if abs(st['rank_ic']) < 0.015:
            return False, 'weak_ic'
        if abs(st['t']) < 2.0:
            return False, 't<2'
        if np.isfinite(c.corr) and c.corr > 0.85:
            return False, 'redundant'
        if strict:
            if composite is None:
                c.incr = abs(st['rank_ic'])
            else:
                c.incr, _ = incremental_ic(c.rank, composite, self.label,
                                           self.masks, self.sample_days)
            if not np.isfinite(c.incr) or abs(c.incr) < 0.002:
                return False, 'no_increment'
        return True, 'pass'

    # -------------------------------------------------------------- 取样
    def _pick_parent(self):
        if len(self.pool) == 1:
            return self.pool[0]
        w = np.array([m.ic ** 1.5 for m in self.pool], float) + 1e-9
        w = w / w.sum()
        r = self.rng.random()
        acc = 0.0
        for m, wi in zip(self.pool, w):
            acc += wi
            if r <= acc:
                return m
        return self.pool[-1]

    def _propose(self, parent, arm):
        if arm in ('harness', 'learning') and len(self.pool) > 1 and self.rng.random() < 0.25:
            return crossover(self.rng, parent.node, self._pick_parent().node)
        return mutate(self.rng, parent.node)

    # -------------------------------------------------------------- 主循环
    def run(self, arm):
        # 每组用独立且确定的随机流, 保证可复现、组间不共享抽样顺序
        self.rng = random.Random(f'{self.seed}-{arm}')
        self.pool, self.cands, self.trace = [], [], []
        self.learn_X, self.learn_y = [], []
        self.seed_stats = []

        # 八组共享同一批初始因子, 每组都付出同样的评价成本
        for name, node in SEEDS:
            c = self.evaluate(node, arm, 0)
            if c.stats is None:
                continue
            c.corr = max_corr(c.rank, [m.rank for m in self.pool]) if self.pool else 0.0
            c.accepted = True
            c.reason = 'seed'
            self.pool.append(c)
            self.seed_stats.append({'name': name, 'formula': expr.to_str(node),
                                    'rank_ic': c.stats['rank_ic'], 't': c.stats['t']})
        if arm == 'baseline':                    # 基线只保留当前最优
            best = max(self.pool, key=lambda c: c.ic)
            for c in self.pool:
                if c is not best:
                    c.accepted, c.reason = False, 'seed_unused'
            self.pool = [best]

        per_round = max(1, self.budget // self.rounds)
        for rnd in range(1, self.rounds + 1):
            composite = self._composite()
            ranks = [m.rank for m in self.pool]
            cands = []

            proposals = []
            n_prop = per_round * 2 if arm == 'learning' else per_round
            for _ in range(n_prop):
                parent = self._pick_parent()
                node = self._propose(parent, arm)
                if expr.nodes(node) > 12:
                    node = parent.node
                proposals.append((node, parent))

            pred_map = {}
            if arm == 'learning' and len(self.learn_X) < 15:
                proposals = proposals[:per_round]
            if arm == 'learning' and len(self.learn_X) >= 15:
                order, preds = self._predict_order(proposals)
                pred_map = {proposals[i][0]: float(preds[i]) for i in range(len(proposals))}
                keep = [proposals[i] for i in order[:per_round]]
                skipped = [proposals[i] for i in order[per_round:]]
                for node, parent in skipped:
                    c = Candidate(self._cid(), arm, rnd, node, parent)
                    c.reason = 'filtered_by_model'
                    c.pred = pred_map.get(node, np.nan)
                    self.cands.append(c)
                proposals = keep

            for node, parent in proposals:
                c = self.evaluate(node, arm, rnd, parent.cid)
                if node in pred_map:
                    c.pred = pred_map[node]
                if c.stats is None:
                    continue
                c.corr = max_corr(c.rank, ranks)
                self.learn_X.append(self._features(node, parent))
                self.learn_y.append(abs(c.stats['rank_ic']))
                cands.append(c)

            # 门控
            for c in cands:
                ok, why = self.gate(c, composite, strict=(arm != 'baseline'))
                c.accepted = ok
                c.reason = why

            if arm == 'baseline':
                best = max(cands, key=lambda c: c.ic) if cands else None
                if best and best.ic > self.pool[0].ic + 1e-6:
                    self.pool = [best]
                    best.accepted, best.reason = True, 'better'
                else:
                    if best:
                        best.accepted, best.reason = False, 'not_better'
                for c in cands:
                    if c is not best:
                        c.accepted = False
                        c.reason = c.reason if c.reason != 'pass' else 'not_selected'
            else:
                for c in sorted([c for c in cands if c.accepted], key=lambda c: -c.ic):
                    if len(self.pool) < self.pool_size:
                        self.pool.append(c)
                    else:
                        worst = min(self.pool, key=lambda m: m.ic)
                        if c.ic > worst.ic + 1e-6:
                            self.pool.remove(worst)
                            worst.accepted, worst.reason = False, 'replaced'
                            self.pool.append(c)
                        else:
                            c.accepted, c.reason = False, 'weaker_than_pool'

            self.trace.append({
                'round': rnd,
                'best_ic': float(max(m.ic for m in self.pool)),
                'pool_mean_ic': float(np.mean([m.ic for m in self.pool])),
                'accepts': int(sum(1 for c in cands if c.accepted)),
                'rejects': {k: sum(1 for c in cands if c.reason == k)
                            for k in ('no_stats', 'coverage', 'weak_ic', 't<2',
                                      'redundant', 'no_increment', 'not_better',
                                      'not_selected', 'weaker_than_pool')
                            if any(c.reason == k for c in cands)},
                'evals': len(cands),
            })

        new_accepts = sum(1 for c in self.cands
                          if c.accepted and c.reason not in ('seed', 'seed_unused'))
        return {'arm': arm, 'evals': len([c for c in self.cands if c.stats]),
                'accepts': sum(1 for c in self.cands if c.accepted and c.reason != 'seed_unused'),
                'accepts_new': new_accepts,
                'pool': self.pool, 'trace': self.trace, 'cands': self.cands,
                'seed_stats': self.seed_stats}

    # -------------------------------------------------------------- 学习型
    def _features(self, node, parent):
        ws = expr.windows(node) or [0]
        subs = expr.subtrees(node)
        return [
            float(expr.nodes(node)), float(expr.depth(node)),
            float(np.mean(ws)), float(max(ws)),
            float(sum(1 for w in ws if w <= 5)),
            1.0 if any(n[0] == 'cs' for n in subs) else 0.0,
            1.0 if any(n[0] == 't2' for n in subs) else 0.0,
            float(abs(parent.stats['rank_ic'])) if parent.stats else 0.0,
        ]

    def _predict_order(self, proposals):
        A = np.asarray(self.learn_X, float)
        y = np.asarray(self.learn_y, float)
        mu, sd = A.mean(0), A.std(0) + 1e-9
        Az = (A - mu) / sd
        ym = y.mean()
        beta = np.linalg.solve(Az.T @ Az + 1e-2 * np.eye(Az.shape[1]), Az.T @ (y - ym))
        X = np.asarray([self._features(n, p) for n, p in proposals], float)
        pred = ((X - mu) / sd) @ beta + ym
        return list(np.argsort(-pred)), pred


def random_candidates(evo: Evolution, n, seed=7):
    """独立的一组随机公式 (不迭代), 用于估计搜索次数的选择偏差."""
    rng = random.Random(seed)
    out = []
    saved = evo.cands
    evo.cands = []
    for _ in range(n):
        c = evo.evaluate(random_tree(rng, 6), 'random', 0)
        if c.stats is not None:
            out.append(c)
    evo.cands = saved
    return out
