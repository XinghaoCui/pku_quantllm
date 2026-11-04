"""因子 -> 可交易账户: 多头前 1/N 等权, 接入带约束的回测引擎.

这一步回答"预测力能不能变成钱": 同样的因子分数, 同样的股票池, 同样的
T+1 / 整手 / 涨跌停 / 容量 / 费用, 与同池等权基准对比.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Backtest


class FactorLong:
    def __init__(self, fac, direction, top=100, exposure=0.98, max_weight=0.05):
        self.fac = fac
        self.d = float(direction)
        self.top = top
        self.exposure = exposure
        self.max_weight = max_weight

    def pool(self, p, sig):
        return p.universe(sig) & np.isfinite(self.fac[sig])

    def target_weights(self, p, sig):
        idx = np.nonzero(self.pool(p, sig))[0]
        if len(idx) == 0:
            return {}
        s = self.d * self.fac[sig, idx]
        order = np.argsort(-s, kind='stable')
        pick = idx[order[:self.top]]
        w = min(self.exposure / len(pick), self.max_weight)
        return {int(j): w for j in pick}


def stats(eq, tr):
    nav = eq['nav'].values
    cum = float(nav[-1] - 1)
    yrs = len(nav) / 243.0
    ann = float((1 + cum) ** (1 / yrs) - 1) if yrs > 0 and cum > -1 else np.nan
    r = pd.Series(nav).pct_change().dropna()
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(243)) if r.std(ddof=1) > 0 else np.nan
    peak = np.maximum.accumulate(nav)
    mdd = float(((nav - peak) / peak).min())
    traded = float(tr['value'].sum()) if len(tr) else 0.0
    cost = float(tr['commission'].sum() + tr['stamp'].sum() +
                 (tr['slippage_cost'] * tr['shares']).sum()) if len(tr) else 0.0
    return {
        'cum_return': cum, 'ann_return': ann, 'sharpe': sharpe, 'mdd': mdd,
        'n_trades': int(len(tr)),
        'turnover_ann': traded / float(eq['equity'].mean()) / yrs,
        'cost': cost,
        'residual_value': float(eq['equity'].iloc[-1] - eq['cash'].iloc[-1]),
        'final_positions': int((eq['n_hold'].iloc[-1])),
    }


def run_factor(p, fac, direction, start, end, top=100, rebal=5, init_cash=1e9,
               commission=1e-4, stamp=5e-4, slippage=5e-4):
    """默认费用: 佣金万一(单边, 最低5元) + 卖出印花税万五 + 单边 5bp 滑点."""
    strat = FactorLong(fac, direction, top=top)
    bt = Backtest(p, strat, start=start, end=end, rebal=rebal, init_cash=init_cash,
                  commission=commission, stamp=stamp, slippage=slippage)
    eq, tr, rj = bt.run()
    st = stats(eq, tr)
    bench = eq['bench'].values
    st['bench_return'] = float(bench[-1] - 1)
    st['excess'] = st['cum_return'] - st['bench_return']
    st['equity'] = eq
    st['trades'] = tr
    st['rejects'] = rj
    return st
