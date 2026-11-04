"""因子评价: IC / RankIC / ICIR + Newey-West + FDR, 五分组, IC 衰减,
训练/验证/测试切分, 代理中性化残差 IC, 因子相关性.

标签口径: t 日收盘出信号 -> t+1 开盘进场 -> t+1+h 开盘出场,
    label_t = O*(t+1+h) / O*(t+1) - 1
避免使用 t+h 收盘这种"隐含当日收盘才能卖出"的口径.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)
SPLITS = {
    'train': ('2021-01-01', '2023-12-31'),
    'valid': ('2024-01-01', '2024-12-31'),
    'test': ('2025-01-01', '2026-09-30'),
}
WARMUP_DAYS = 252


# ----------------------------------------------------------------- 标签
def fwd_labels(p, h: int) -> np.ndarray:
    """O*(t+1+h)/O*(t+1)-1, 与 t 日信号对齐."""
    n = p.opstar.shape[0]
    out = np.full_like(p.opstar, np.nan)
    if n > h + 1:
        out[:n - h - 1] = p.opstar[h + 1:] / p.opstar[1:n - h] - 1.0
    return out.astype(np.float32)


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method='average').values


def _corr(x, y, min_n=30):
    if len(x) < min_n:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return np.nan
    return float((x * y).mean() / (sx * sy))


# --------------------------------------------------------------- 显著性
def nw_t(x, lag: int = 5):
    """Newey-West (Bartlett) t 统计量, 返回 (t, 双侧 p)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    m = x.mean()
    d = x - m
    var = float(d @ d) / n
    for l in range(1, min(lag, n - 1) + 1):
        g = float(d[l:] @ d[:-l]) / n
        var += 2.0 * (1.0 - l / (lag + 1.0)) * g
    var = max(var, 1e-18)
    t = m / np.sqrt(var / n)
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))          # 正态近似双侧 p
    return float(t), float(p)


def bh_fdr(pvals):
    """Benjamini-Hochberg q 值."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        val = min(prev, p[order[k]] * m / (k + 1))
        q[order[k]] = val
        prev = val
    return q


# --------------------------------------------------------------- 单日截面
def cross_section(fac, lab, mask):
    """单日: 返回 (ic, rank_ic, n, group_mean(q5), decile_boundaries)."""
    m = mask & np.isfinite(fac) & np.isfinite(lab)
    n = int(m.sum())
    if n < 50:
        return None
    x = fac[m].astype(float)
    y = lab[m].astype(float)
    ic = _corr(x, y)
    rx, ry = _rank(x), _rank(y)
    ric = _corr(rx, ry)
    groups = np.array_split(np.argsort(rx, kind='stable'), 5)
    gm = np.array([y[g].mean() for g in groups])
    return ic, ric, n, gm


def factor_stats(fac, lab, mask, days, h: int):
    ics, rics, ns, gms, used = [], [], [], [], []
    for t in days:
        if not (np.isfinite(fac[t]).any() and np.isfinite(lab[t]).any()):
            continue
        r = cross_section(fac[t], lab[t], mask[t])
        if r is None:
            continue
        ics.append(r[0]); rics.append(r[1]); ns.append(r[2]); gms.append(r[3]); used.append(t)
    ics = np.asarray(ics, float); rics = np.asarray(rics, float)
    gms = np.asarray(gms, float) if gms else np.zeros((0, 5))
    t_stat, p = nw_t(rics, lag=max(h, 3))
    mono = np.nan
    if len(gms) > 10:
        mono = _corr(np.arange(5.0, dtype=float), gms.mean(axis=0), min_n=3)
    return {
        'n_days': int(len(rics)),
        'ic_mean': float(np.nanmean(ics)) if len(ics) else np.nan,
        'rank_ic_mean': float(np.nanmean(rics)) if len(rics) else np.nan,
        'rank_ic_std': float(np.nanstd(rics, ddof=1)) if len(rics) > 1 else np.nan,
        'icir': float(np.nanmean(rics) / np.nanstd(rics, ddof=1)) if len(rics) > 1 and np.nanstd(rics, ddof=1) > 0 else np.nan,
        't_hac': t_stat, 'p_hac': p,
        'pos_ratio': float(np.nanmean(rics > 0)) if len(rics) else np.nan,
        'avg_names': float(np.mean(ns)) if ns else np.nan,
        'quantiles': gms.mean(axis=0).tolist() if len(gms) else [],
        'monotonic': float(mono) if np.isfinite(mono) else np.nan,
        'ls_mean': float(gms[:, -1].mean() - gms[:, 0].mean()) if len(gms) else np.nan,
        'ls_sharpe': _ls_sharpe(gms, h) if len(gms) > 10 else np.nan,
        '_series': (ics, rics), '_days': np.asarray(used, dtype=int),
    }


def _ls_sharpe(gms, h):
    ls = gms[:, -1] - gms[:, 0]
    if len(ls) < 10 or ls.std() == 0:
        return np.nan
    return float(ls.mean() / ls.std(ddof=1) * np.sqrt(243.0 / h))


# ------------------------------------------------------------ 中性化残差
def residual_rank_ic(fac, lab, p, mask, days, lag=5):
    """对 [log 价格, log 成交额, 20日波动, 20日动量] 的横截面秩做 OLS,
    取残差与标签秩相关. 只是有限代理控制, 不是行业/市值中性化.

    同一批日期上同时给出原始 RankIC, 便于看"残差还剩下多少".
    """
    amt20 = p.roll('amt', 20, 'mean')
    vol20 = p.roll('ret', 20, 'std')
    with np.errstate(invalid='ignore', divide='ignore'):
        lp = np.log(p.close)
        mo = p.pstar / np.vstack([np.full((20, p.pstar.shape[1]), np.nan), p.pstar[:-20]]) - 1.0
    expo = [lp, np.log1p(amt20), vol20, mo]
    raw, res = [], []
    for t in days:
        m = mask[t] & np.isfinite(fac[t]) & np.isfinite(lab[t])
        if m.sum() < 100:
            continue
        cols = [_rank(e[t][m]) for e in expo]
        if any(np.nanstd(c) == 0 for c in cols):
            continue
        y = _rank(fac[t][m])
        ry = _rank(lab[t][m])
        raw.append(_corr(y, ry))
        X = np.column_stack([np.ones(m.sum())] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res.append(_corr(y - X @ beta, ry))
    raw = np.asarray([v for v in raw if np.isfinite(v)])
    res = np.asarray([v for v in res if np.isfinite(v)])
    if len(res) < 10:
        return {'raw_rankin': np.nan, 'resid_rankin': np.nan, 'resid_t': np.nan, 'n': int(len(res))}
    t_res, _ = nw_t(res, lag=lag)
    return {'raw_rankin': float(raw.mean()), 'resid_rankin': float(res.mean()),
            'resid_t': t_res, 'n': int(len(res))}
