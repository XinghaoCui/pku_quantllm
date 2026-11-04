"""行情面板与可交易性定义.

七个 Feather 宽表 -> numpy 矩阵 (n_days x n_codes).
  P*  = close * cumadj   总收益价格 (含分红送转)
  O*  = open  * cumadj   开盘总收益价格, 用于"次日开盘入场"的标签

ST 状态来自两个来源, 面板会记录每格来自哪一个:
  0 官方 (BaoStock 逐日 isST)
  1 价格行为推断 (±5% 涨跌幅, 详见 build_st_proxy.py)
未覆盖 -> 视为未知, 禁止买入.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

MATRICES = ['open', 'high', 'low', 'close', 'amt', 'num_trades', 'cumadj']


def half_up(x):
    """交易所四舍五入到分 (half-up). numpy 的 round 是银行家舍入, 会算错涨跌停价."""
    return np.floor(np.asarray(x, dtype=float) * 100.0 + 0.5) / 100.0


def limit_pct(codes) -> np.ndarray:
    out = np.full(len(codes), 0.10)
    for k, c in enumerate(codes):
        if c[:3] in ('688', '689', '300', '301'):
            out[k] = 0.20
    return out


class Panel:
    def __init__(self, data_dir='feathers', st_path=None, st_proxy_path=None):
        raw = {}
        for k in MATRICES:
            df = pd.read_feather(os.path.join(data_dir, f'{k}.feather')).set_index('date')
            raw[k] = df.astype('float64')
        self.dates = pd.DatetimeIndex(raw['close'].index)
        self.codes = list(raw['close'].columns)
        self._code_idx = {c: j for j, c in enumerate(self.codes)}
        # copy=True: pandas 3.0 的 .values 可能返回只读视图
        def arr(k):
            return raw[k].to_numpy(dtype=float, copy=True)
        self.open = arr('open')
        self.high = arr('high')
        self.low = arr('low')
        self.close = arr('close')
        self.amt = arr('amt')
        self.num_trades = arr('num_trades')
        self.cumadj = arr('cumadj')
        self.pstar = self.close * self.cumadj
        self.opstar = self.open * self.cumadj

        self._load_st(st_path, st_proxy_path)
        self._build_tradability()
        self._build_features()

    # ------------------------------------------------------------------ ST
    def _load_st(self, st_path, proxy_path):
        nD, nC = self.close.shape
        self.st = np.full((nD, nC), -1, dtype=np.int8)
        self.st_src = np.full(nC, -1, dtype=np.int8)      # 0 官方 / 1 推断 / -1 未知
        self.st_stats = {'official': 0, 'proxy': 0, 'unknown': 0, 'n_codes': nC,
                         'proxy_filled_cells': 0}
        if st_path and os.path.exists(st_path):
            m = pd.read_parquet(st_path).reindex(index=self.dates)
            cols = [c for c in self.codes if c in m.columns]
            pos = [self._code_idx[c] for c in cols]
            self.st[:, pos] = np.nan_to_num(m[cols].values, nan=-1).astype(np.int8)
            self.st_src[pos] = 0
            self.st_stats['official'] = len(cols)
        if proxy_path and os.path.exists(proxy_path):
            m = pd.read_parquet(proxy_path).reindex(index=self.dates)
            cols = [c for c in self.codes if c in m.columns]
            pos = [self._code_idx[c] for c in cols]
            prx = np.full((nD, nC), -1, dtype=np.int8)
            prx[:, pos] = np.nan_to_num(m[cols].values, nan=-1).astype(np.int8)
            # 只在官方标签缺失的格子上用推断值填补 (官方 2024 年起才有)
            gap = (self.st < 0) & (prx >= 0)
            self.st_stats['proxy_filled_cells'] = int(gap.sum())
            self.st[gap] = prx[gap]
            no_official = self.st_src < 0
            self.st_src[no_official & (prx >= 0).any(axis=0)] = 1
            self.st_stats['proxy'] = len(cols)
        self.st_stats['unknown'] = int((self.st_src < 0).sum())

    # --------------------------------------------------------- 可交易性
    def _build_tradability(self):
        nD, nC = self.close.shape
        prev = np.vstack([np.full((1, nC), np.nan), self.close[:-1]])
        lp = limit_pct(self.codes)[None, :]
        st5 = np.where(self.st == 1, 0.05, lp)
        self.limit_up = half_up(prev * (1.0 + st5))
        self.limit_dn = half_up(prev * (1.0 - st5))
        self.has_prev = np.isfinite(prev) & (prev > 0)
        self.suspended = ~(np.isfinite(self.open) & (self.open > 0))
        self.hist = pd.DataFrame(np.isfinite(self.close), index=self.dates,
                                 columns=self.codes).cumsum().values

    def _build_features(self):
        ret = np.full_like(self.pstar, np.nan)
        ret[1:] = self.pstar[1:] / self.pstar[:-1] - 1.0
        self.ret = ret
        self._ret = pd.DataFrame(ret, index=self.dates, columns=self.codes)
        self._amt = pd.DataFrame(self.amt, index=self.dates, columns=self.codes)
        self._ntr = pd.DataFrame(self.num_trades, index=self.dates, columns=self.codes)
        self._cache = {}

    def roll(self, src: str, win: int, kind: str) -> np.ndarray:
        key = (src, win, kind)
        if key not in self._cache:
            df = {'ret': self._ret, 'amt': self._amt, 'ntr': self._ntr}[src]
            self._cache[key] = getattr(df.rolling(win, min_periods=max(5, win // 2)), kind)().values
        return self._cache[key]

    # -------------------------------------------------------------- 工具
    def idx(self, ts) -> int:
        return int(np.searchsorted(self.dates, pd.Timestamp(ts)))

    def last_le(self, ts) -> int:
        return int(np.searchsorted(self.dates, pd.Timestamp(ts), side='right')) - 1

    def slice(self, start, end):
        return slice(self.idx(start), self.last_le(end) + 1)

    def universe(self, sig: int, min_hist=252, min_amt20=2e7) -> np.ndarray:
        """因子评价的横截面股票池: 非 ST(含未知剔除) + 有价 + 上市够久 + 流动性."""
        amt20 = self.roll('amt', 20, 'mean')[sig]
        ok = ((self.st[sig] == 0)
              & np.isfinite(self.close[sig]) & (self.close[sig] > 0)
              & (self.hist[sig] >= min_hist)
              & np.isfinite(amt20) & (amt20 >= min_amt20))
        return ok
