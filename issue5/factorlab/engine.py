"""精简撮合引擎: 收盘出信号 -> 次日开盘成交.

总收益份额记账:  units_j = 股数 / cumadj,  市值 = units * close * cumadj.
买入时 units += 股数/cumadj, 因此滑点、佣金都真实体现在净值里.

规则: T+1、整手 100 股、涨跌停拒单 (交易所 half-up 口径)、停牌跳过、
单日单票成交额 <= 过去 20 日均额 x 参与率、佣金单边万一起、卖出印花税万五、
期末按日分批退出, 卖不掉的残仓按收盘价计入净值并单独披露.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Book:
    def __init__(self, panel, init_cash=1e9, commission=1e-4, min_commission=5.0,
                 stamp=5e-4, slippage=5e-4, participation=0.05, lot=100):
        self.p = panel
        self.cash = float(init_cash)
        self.init_cash = float(init_cash)
        self.commission = commission
        self.min_commission = min_commission
        self.stamp = stamp
        self.slippage = slippage
        self.participation = participation
        self.lot = lot
        self.units = np.zeros(len(panel.codes))
        self.last_buy = np.full(len(panel.codes), -10**9, dtype=int)
        self.trades = []
        self.rejects = []
        self._amt20 = panel.roll('amt', 20, 'mean')

    # ------------------------------------------------------------- 估值
    def value(self, i):
        return float(self.cash + np.nansum(self.units * self.p.pstar[i]))

    def _pos_value(self, sig, j):
        v = self.units[j] * self.p.pstar[sig, j]
        return v if np.isfinite(v) else 0.0

    def _reject(self, i, j, side, want, reason):
        if want >= 1e4:
            self.rejects.append((self.p.dates[i], self.p.codes[j], side, float(want), reason))

    # ------------------------------------------------------------- 成交
    def buy(self, i, sig, j, want_value):
        if self.p.st[i, j] == 1:          # 成交日已是 ST, 不允许建仓
            self._reject(i, j, 'BUY', want_value, 'st'); return 0.0
        px0 = self.p.open[i, j]
        if not np.isfinite(px0) or px0 <= 0:
            self._reject(i, j, 'BUY', want_value, 'suspended'); return 0.0
        if px0 >= self.p.limit_up[i, j] - 1e-9:
            self._reject(i, j, 'BUY', want_value, 'limit_up'); return 0.0
        px = px0 * (1 + self.slippage)
        a20 = self._amt20[sig, j]
        cap = self.participation * a20 if np.isfinite(a20) and a20 > 0 else 0.0
        want = min(want_value, cap)
        if want <= 0:
            self._reject(i, j, 'BUY', want_value, 'no_capacity'); return 0.0
        sh = int(want // px // self.lot) * self.lot
        sh = min(sh, int(self.cash / (px * (1 + self.commission)) // self.lot) * self.lot)
        if sh < self.lot:
            self._reject(i, j, 'BUY', want_value, 'cash' if sh <= 0 else 'below_lot'); return 0.0
        gross = sh * px
        comm = max(gross * self.commission, self.min_commission)
        if gross + comm > self.cash + 1e-6:
            sh -= self.lot
            if sh < self.lot:
                self._reject(i, j, 'BUY', want_value, 'cash'); return 0.0
            gross = sh * px; comm = max(gross * self.commission, self.min_commission)
        self.cash -= gross + comm
        self.units[j] += sh / self.p.cumadj[i, j]
        self.last_buy[j] = i
        self.trades.append((self.p.dates[i], self.p.codes[j], 'BUY', sh, px, gross,
                            comm, 0.0, px - px0, 'rebalance'))
        return gross

    def sell(self, i, sig, j, want_value, full=False, reason='rebalance'):
        px0 = self.p.open[i, j]
        held = self.units[j] * self.p.cumadj[i, j]
        if held <= 0:
            return 0.0
        if self.last_buy[j] >= i:
            self._reject(i, j, 'SELL', want_value, 't_plus_1'); return 0.0
        if not np.isfinite(px0) or px0 <= 0:
            self._reject(i, j, 'SELL', want_value, 'suspended'); return 0.0
        if px0 <= self.p.limit_dn[i, j] + 1e-9:
            self._reject(i, j, 'SELL', want_value, 'limit_down'); return 0.0
        px = px0 * (1 - self.slippage)
        a20 = self._amt20[sig, j]
        cap_sh = int(self.participation * a20 / px // self.lot) * self.lot if np.isfinite(a20) and a20 > 0 else 0
        want_sh = min(held if full else want_value / px, held, cap_sh)
        whole = full and want_sh >= held - 1e-9
        sh = float(held) if whole else float(int(want_sh // self.lot) * self.lot)
        if sh <= 0:
            self._reject(i, j, 'SELL', want_value,
                         'no_capacity' if cap_sh <= 0 else 'below_lot'); return 0.0
        gross = sh * px
        comm = max(gross * self.commission, self.min_commission)
        stamp = gross * self.stamp
        self.cash += gross - comm - stamp
        self.units[j] -= sh / self.p.cumadj[i, j]
        if self.units[j] < 1e-9:
            self.units[j] = 0.0
        self.trades.append((self.p.dates[i], self.p.codes[j], 'SELL', sh, px, gross,
                            comm, stamp, px0 - px, reason))
        return gross


class Backtest:
    """通用日频回测: 由 strategy.target_weights(panel, sig, cfg) 给出目标权重."""

    def __init__(self, panel, strategy, start='2025-01-02', end='2026-09-30',
                 rebal=5, exit_days=5, init_cash=1e9, cost_bps=10.0, **book_kw):
        self.p = panel
        self.s = strategy
        self.i0 = panel.idx(start)
        self.iL = panel.last_le(end)
        self.rebal = rebal
        self.exit_start = max(self.i0 + 1, self.iL - exit_days + 1)
        self.rebal_days = set(range(self.i0, self.exit_start, rebal))
        self.book = Book(panel, init_cash=init_cash, **book_kw)
        self.cash0 = init_cash
        self.cost_bps = cost_bps
        self.equity = []

    def run(self):
        b, p = self.book, self.p
        for i in range(self.i0, self.iL + 1):
            sig = i - 1
            in_exit = i >= self.exit_start
            if i in self.rebal_days or in_exit:
                targets = {} if in_exit else self.s.target_weights(p, sig)
                self._rebalance(i, sig, targets, in_exit)
            eq = b.value(i)
            self.equity.append((p.dates[i], eq, b.cash,
                                int(np.count_nonzero(b.units * np.nan_to_num(p.pstar[i]) > 1.0)),
                                eq / self.cash0))
        eq = pd.DataFrame(self.equity, columns=['date', 'equity', 'cash', 'n_hold', 'nav'])
        eq['bench'] = self._benchmark()
        cols = ['date', 'code', 'side', 'shares', 'price', 'value', 'commission',
                'stamp', 'slippage_cost', 'reason']
        tr = pd.DataFrame(b.trades, columns=cols)
        rj = pd.DataFrame(b.rejects, columns=['date', 'code', 'side', 'want_value', 'reason'])
        return eq, tr, rj

    def _rebalance(self, i, sig, targets, in_exit):
        b = self.book
        eq_sig = b.value(sig)
        for j in np.nonzero(b.units > 0)[0]:
            j = int(j)
            cur = b._pos_value(sig, j)
            if cur <= 0:
                continue
            if j not in targets:
                b.sell(i, sig, j, cur, full=True, reason='exit' if in_exit else 'drop')
                continue
            tgt = targets[j] * eq_sig
            if cur - tgt > max(1e-6 * eq_sig, 1.0):
                b.sell(i, sig, j, cur - tgt, reason='rebalance')
        if in_exit:
            return
        for j, w in sorted(targets.items(), key=lambda kv: -kv[1]):
            j = int(j)
            delta = w * eq_sig - b._pos_value(sig, j)
            if delta > 0:
                b.buy(i, sig, j, delta)

    def _benchmark(self):
        """同池等权、零成本、同一调仓日."""
        p = self.p
        nav = np.ones(self.iL - self.i0 + 1)
        cur = 1.0
        days = range(self.i0, self.iL + 1)
        for i in days:
            nav[i - self.i0] = cur
            if i >= self.exit_start or i not in self.rebal_days:
                continue
            sig = i - 1
            pool = np.nonzero(self.s.pool(p, sig))[0]
            if len(pool) == 0:
                continue
            nxt = min([d for d in sorted(self.rebal_days) if d > i] + [self.exit_start])
            for d in range(i + 1, min(nxt, self.iL) + 1):
                r = np.nanmean(p.pstar[d, pool] / p.pstar[d - 1, pool] - 1.0)
                cur *= 1.0 + (0.0 if not np.isfinite(r) else r)
                nav[d - self.i0] = cur
        return nav
