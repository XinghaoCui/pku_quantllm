"""因子库: 每个因子返回 (n_days, n_codes) float32 矩阵.

约定: 因子在 t 日收盘后可用, 只使用 t 及之前的数据 (向后 shift 不会出现负位移).
数值越大代表"越看好"(经济先验方向已写入符号), 因子方向最终仍由训练期
RankIC 的符号冻结, 测试期不翻转.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """取 t-k 的值 (k>0 表示过去)."""
    out = np.full_like(a, np.nan)
    if k <= 0:
        return a
    out[k:] = a[:-k]
    return out


def _df(a, p):
    return pd.DataFrame(a, index=p.dates, columns=p.codes)


def log_ret(p: Panel, n: int) -> np.ndarray:
    with np.errstate(invalid='ignore', divide='ignore'):
        return p.pstar / _shift(p.pstar, n) - 1.0


# ------------------------------------------------------------------ 因子
def rev5(p: Panel, **kw):
    return -log_ret(p, 5)


def rev20(p: Panel, **kw):
    return -log_ret(p, 20)


def mom20(p: Panel, **kw):
    return log_ret(p, 20)


def mom60_skip5(p: Panel, **kw):
    """t-5 相对 t-65 的收益, 跳过最近 5 日."""
    a = _shift(p.pstar, 5)
    b = _shift(p.pstar, 65)
    with np.errstate(invalid='ignore', divide='ignore'):
        return a / b - 1.0


def lowvol20(p: Panel, **kw):
    return -p.roll('ret', 20, 'std')


def lowvol60(p: Panel, **kw):
    return -p.roll('ret', 60, 'std')


def lowamt20(p: Panel, **kw):
    return -np.log1p(p.roll('amt', 20, 'mean'))


def lowntr20(p: Panel, **kw):
    return -np.log1p(p.roll('ntr', 20, 'mean'))


def amt_shrink(p: Panel, **kw):
    """近期成交额相对 20 日均值的收缩幅度, 越大越冷门."""
    a5 = p.roll('amt', 5, 'mean')
    a20 = p.roll('amt', 20, 'mean')
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - a5 / a20


def intraday_rev(p: Panel, **kw):
    """5 日平均日内涨幅取负 (日内跌得多 -> 看多)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        x = p.close / p.open - 1.0
    return -_df(x, p).rolling(5, min_periods=3).mean().values


def overnight_rev(p: Panel, **kw):
    """5 日平均隔夜跳空取负."""
    with np.errstate(invalid='ignore', divide='ignore'):
        x = p.open / _shift(p.close, 1) - 1.0
    return -_df(x, p).rolling(5, min_periods=3).mean().values


def range_low20(p: Panel, **kw):
    """收盘在 20 日高低区间中的位置取负 (越低位越看多)."""
    hi = _df(p.pstar, p).rolling(20, min_periods=10).max().values
    lo = _df(p.pstar, p).rolling(20, min_periods=10).min().values
    with np.errstate(invalid='ignore', divide='ignore'):
        pos = (p.pstar - lo) / (hi - lo)
    return -pos


def rsi_rev14(p: Panel, **kw):
    """14 日 RSI 取负: 超卖看多."""
    up = _df(np.where(p.ret > 0, p.ret, 0.0), p).rolling(14, min_periods=10).mean()
    dn = _df(np.where(p.ret < 0, -p.ret, 0.0), p).rolling(14, min_periods=10).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        rsi = up / (up + dn)
    return -rsi.values


REGISTRY = {
    'rev5': (rev5, '5日反转', '短期反转', '-P*(t)/P*(t-5)+1'),
    'rev20': (rev20, '20日反转', '反转', '-P*(t)/P*(t-20)+1'),
    'mom20': (mom20, '20日动量', '动量', 'P*(t)/P*(t-20)-1'),
    'mom60_skip5': (mom60_skip5, '60日动量(跳过5日)', '动量', 'P*(t-5)/P*(t-65)-1'),
    'lowvol20': (lowvol20, '20日低波动', '风险', '-std(r_t, 20)'),
    'lowvol60': (lowvol60, '60日低波动', '风险', '-std(r_t, 60)'),
    'lowamt20': (lowamt20, '低成交额', '流动性', '-log(ADV20)'),
    'lowntr20': (lowntr20, '低成交笔数', '流动性', '-log(笔数20)'),
    'amt_shrink': (amt_shrink, '成交额收缩', '流动性', '1 - ADV5/ADV20'),
    'intraday_rev': (intraday_rev, '日内反转', '反转', '-mean(close/open-1, 5)'),
    'overnight_rev': (overnight_rev, '隔夜反转', '反转', '-mean(open/prev_close-1, 5)'),
    'range_low20': (range_low20, '20日区间低位', '反转', '-(P*-min20)/(max20-min20)'),
    'rsi_rev14': (rsi_rev14, 'RSI14 反转', '反转', '-RSI14'),
}


def build(key: str, p: Panel) -> np.ndarray:
    fn = REGISTRY[key][0]
    return np.asarray(fn(p), dtype=np.float32)


def catalog() -> list[dict]:
    return [{'key': k, 'name': v[1], 'category': v[2], 'formula': v[3]}
            for k, v in REGISTRY.items()]
