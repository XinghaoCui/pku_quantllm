"""候选评分: RankIC / HAC t / 分组 / 与因子池的相关与增量."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import evaluate as E


def score_candidate(fac, label, masks, days, h=5):
    """在给定交易日上计算横截面 RankIC 序列与分组收益."""
    if np.ndim(fac) != 2:
        return None
    ics, rics, gms, ns, used = [], [], [], [], []
    for t in days:
        m = masks[t] & np.isfinite(fac[t]) & np.isfinite(label[t])
        n = int(m.sum())
        if n < 200:
            continue
        x = fac[t][m].astype(float)
        y = label[t][m].astype(float)
        rx = pd.Series(x).rank(method='average').values
        ry = pd.Series(y).rank(method='average').values
        ics.append(E._corr(x, y, min_n=5))
        rics.append(E._corr(rx, ry, min_n=5))
        groups = np.array_split(np.argsort(rx, kind='stable'), 5)
        gms.append(np.array([y[g].mean() for g in groups]))
        ns.append(n)
        used.append(t)
    ics = np.asarray(ics, float)
    rics = np.asarray(rics, float)
    gms = np.asarray(gms, float) if gms else np.zeros((0, 5))
    ok = np.isfinite(rics)
    ics, rics = ics[ok], rics[ok]
    if len(rics) < 30:
        return None
    t, p = E.nw_t(rics, lag=max(h, 3))
    std = rics.std(ddof=1)
    return {
        'n_days': int(len(rics)),
        'rank_ic': float(rics.mean()),
        'ic': float(ics.mean()),
        'std': float(std),
        'icir': float(rics.mean() / std) if std > 0 else np.nan,
        't': t, 'p': p,
        'pos': float((rics > 0).mean()),
        'avg_names': float(np.mean(ns)) if ns else np.nan,
        'quantiles': gms.mean(axis=0).tolist() if len(gms) else [],
        'mono': float(E._corr(np.arange(5.0), gms.mean(axis=0), min_n=3)) if len(gms) > 10 else np.nan,
        'ls': float(gms[:, -1].mean() - gms[:, 0].mean()) if len(gms) else np.nan,
        '_series': rics,
    }


def sampled_ranks(fac, masks, days):
    """抽样日上的截面秩, 用于计算因子之间的相关 (n_sample x n_codes)."""
    R = np.full((len(days), len(masks[0])), np.nan, dtype=np.float32)
    for k, t in enumerate(days):
        m = masks[t] & np.isfinite(fac[t])
        if m.sum() > 100:
            R[k, m] = pd.Series(fac[t][m]).rank().values.astype(np.float32)
    return R


def max_corr(R, member_ranks):
    best = 0.0
    for other in member_ranks:
        cs = []
        for r in range(R.shape[0]):
            ok = np.isfinite(R[r]) & np.isfinite(other[r])
            if ok.sum() > 100:
                v = E._corr(R[r][ok], other[r][ok], min_n=50)
                if np.isfinite(v):
                    cs.append(v)
        if cs:
            best = max(best, abs(float(np.mean(cs))))
    return best


def incremental_ic(R, composite_ranks, label, masks, days):
    """剔除池组合后剩余的 RankIC (单变量回归残差).

    只用抽样日, 目的是快速门控; 报告里注明这是抽样估计.
    """
    vals = []
    for k, t in enumerate(days):
        m = masks[t] & np.isfinite(R[k]) & np.isfinite(composite_ranks[k]) & np.isfinite(label[t])
        if m.sum() < 200:
            continue
        x = R[k][m].astype(float)
        c = composite_ranks[k][m].astype(float)
        y = pd.Series(label[t][m]).rank().values
        c = c - c.mean()
        denom = float(c @ c)
        beta = float(c @ x) / denom if denom > 1e-12 else 0.0
        resid = x - beta * c
        v = E._corr(resid, y, min_n=50)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)
