"""因子表达式 DSL.

表达式是不可变嵌套元组, 可直接做 dict key (便于共享子表达式):

    ('v',  'amt')                      变量
    ('c',  1.0)                        常数
    ('u',  'neg',  child)              一元
    ('b',  'div',  a, b)               二元
    ('t',  'mean', child, window)      时序算子
    ('t2', 'corr', a, b, window)       双序列时序算子
    ('cs', 'rank', child)              横截面算子

约定: 所有算子只使用 t 及之前的数据 (rolling 右对齐), 不会引入未来信息.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

WINDOWS = (3, 5, 10, 20, 60)

TS_UNARY = ('mean', 'std', 'sum', 'min', 'max', 'delta', 'pct', 'zscore', 'delay')
TS_BINARY = ('corr', 'cov')
UNARY = ('neg', 'abs', 'log', 'sqrt_abs', 'sign')
BINARY = ('add', 'sub', 'mul', 'div', 'max', 'min')
CS = ('rank', 'zscore', 'demean')
VARS = ('pstar', 'ret', 'intraday', 'overnight', 'hl', 'amt', 'ntr',
        'amt_pt', 'close', 'open', 'high', 'low')


# ------------------------------------------------------------------ 工具
def nodes(n) -> int:
    k = n[0]
    if k in ('v', 'c'):
        return 1
    if k in ('u', 'cs'):
        return 1 + nodes(n[2])
    if k == 'b':
        return 1 + nodes(n[2]) + nodes(n[3])
    if k == 't':
        return 1 + nodes(n[2])
    if k == 't2':
        return 1 + nodes(n[2]) + nodes(n[3])
    raise ValueError(n)


def depth(n) -> int:
    k = n[0]
    if k in ('v', 'c'):
        return 1
    if k in ('u', 'cs'):
        return 1 + depth(n[2])
    if k == 'b':
        return 1 + max(depth(n[2]), depth(n[3]))
    if k == 't':
        return 1 + depth(n[2])
    if k == 't2':
        return 1 + max(depth(n[2]), depth(n[3]))
    raise ValueError(n)


def windows(n) -> list:
    k = n[0]
    if k in ('v', 'c'):
        return []
    if k == 't':
        return [n[3]] + windows(n[2])
    if k == 't2':
        return [n[4]] + windows(n[2]) + windows(n[3])
    if k in ('u', 'cs'):
        return windows(n[2])
    return windows(n[2]) + windows(n[3])


def to_str(n) -> str:
    k = n[0]
    if k == 'v':
        return n[1]
    if k == 'c':
        return f'{n[1]:g}'
    if k == 'u':
        return f'{n[1]}({to_str(n[2])})'
    if k == 'b':
        sym = {'add': '+', 'sub': '-', 'mul': '×', 'div': '/', 'max': 'max', 'min': 'min'}[n[1]]
        if n[1] in ('max', 'min'):
            return f'{n[1]}({to_str(n[2])}, {to_str(n[3])})'
        return f'({to_str(n[2])} {sym} {to_str(n[3])})'
    if k == 't':
        return f'{n[1]}({to_str(n[2])}, {n[3]})'
    if k == 't2':
        return f'{n[1]}({to_str(n[2])}, {to_str(n[3])}, {n[4]})'
    if k == 'cs':
        return f'{n[1]}({to_str(n[2])})'
    raise ValueError(n)


def digest(n) -> str:
    return hashlib.sha1(json.dumps(n).encode()).hexdigest()[:10]


def subtrees(n, acc=None):
    """返回 [(路径, 子树)], 路径用于替换. 路径是 int 列表."""
    acc = [] if acc is None else acc
    acc.append(n)
    k = n[0]
    if k in ('u', 'cs'):
        subtrees(n[2], acc)
    elif k == 'b':
        subtrees(n[2], acc); subtrees(n[3], acc)
    elif k == 't':
        subtrees(n[2], acc)
    elif k == 't2':
        subtrees(n[2], acc); subtrees(n[3], acc)
    return acc


# ------------------------------------------------------------------ 求值
class Ctx:
    """一次评估会话: 持有行情矩阵和变量缓存."""

    def __init__(self, panel):
        self.p = panel
        self.shape = panel.pstar.shape
        self.df = lambda a: pd.DataFrame(a, index=panel.dates, columns=panel.codes)
        self.vars = {}
        p = panel
        self.vars['pstar'] = p.pstar
        self.vars['ret'] = p.ret
        with np.errstate(invalid='ignore', divide='ignore'):
            self.vars['intraday'] = p.close / p.open - 1.0
            prev = np.vstack([np.full((1, p.close.shape[1]), np.nan), p.close[:-1]])
            self.vars['overnight'] = p.open / prev - 1.0
            self.vars['hl'] = p.high / p.low - 1.0
            self.vars['amt_pt'] = p.amt / np.maximum(p.num_trades, 1.0)
        self.vars['amt'] = p.amt
        self.vars['ntr'] = p.num_trades
        self.vars['close'] = p.close
        self.vars['open'] = p.open
        self.vars['high'] = p.high
        self.vars['low'] = p.low

    def evaluate(self, node):
        """顶层求值: 常数也要广播成完整矩阵."""
        out = self.eval(node)
        if np.ndim(out) == 0:
            out = np.full(self.shape, out, dtype=np.float32)
        return out

    def eval(self, node, cache=None):
        cache = {} if cache is None else cache
        if node in cache:
            return cache[node]
        k = node[0]
        if k == 'v':
            out = self.vars[node[1]]
        elif k == 'c':
            out = np.float32(node[1])
        elif k == 'u':
            x = self.eval(node[2], cache)
            op = node[1]
            if op == 'neg':
                out = -x
            elif op == 'abs':
                out = np.abs(x)
            elif op == 'log':
                out = np.log(np.maximum(x, 1e-9))
            elif op == 'sqrt_abs':
                out = np.sign(x) * np.sqrt(np.abs(x))
            elif op == 'sign':
                out = np.sign(x)
            else:
                raise ValueError(op)
        elif k == 'b':
            a = self.eval(node[2], cache)
            b = self.eval(node[3], cache)
            out = _binary(node[1], a, b)
        elif k == 't':
            x = self.eval(node[2], cache)
            out = _ts_unary(node[1], x, node[3], self)
        elif k == 't2':
            a = self.eval(node[2], cache)
            b = self.eval(node[3], cache)
            out = _ts_binary(node[1], a, b, node[4], self)
        elif k == 'cs':
            x = self.eval(node[2], cache)
            out = _cs(node[1], x, self)
        else:
            raise ValueError(node)
        out = np.asarray(out, dtype=np.float32)
        cache[node] = out
        return out


def _arr(x, shape):
    return np.full(shape, x, dtype=np.float32) if np.isscalar(x) else x


def _binary(op, a, b):
    shape = a.shape if np.ndim(a) == 2 else b.shape
    a = _arr(a, shape); b = _arr(b, shape)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        if op == 'add':
            out = a + b
        elif op == 'sub':
            out = a - b
        elif op == 'mul':
            out = a * b
        elif op == 'div':
            out = a / np.where(np.abs(b) < 1e-9, np.nan, b)
        elif op == 'max':
            out = np.maximum(a, b)
        elif op == 'min':
            out = np.minimum(a, b)
        else:
            raise ValueError(op)
    return out


def _ts_unary(op, x, w, ctx):
    if np.ndim(x) == 0:
        return np.full(ctx.p.pstar.shape, x, dtype=np.float32)
    if op == 'delay':
        out = np.full_like(x, np.nan)
        out[w:] = x[:-w]
        return out
    df = ctx.df(x)
    if op == 'mean':
        r = df.rolling(w, min_periods=max(2, w // 2)).mean()
    elif op == 'std':
        r = df.rolling(w, min_periods=max(3, w // 2)).std()
    elif op == 'sum':
        r = df.rolling(w, min_periods=max(2, w // 2)).sum()
    elif op == 'min':
        r = df.rolling(w, min_periods=max(2, w // 2)).min()
    elif op == 'max':
        r = df.rolling(w, min_periods=max(2, w // 2)).max()
    elif op == 'pct':
        prev = np.full_like(x, np.nan)
        prev[w:] = x[:-w]
        with np.errstate(invalid='ignore', divide='ignore'):
            return x / np.where(np.abs(prev) < 1e-12, np.nan, prev) - 1.0
    elif op == 'delta':
        prev = np.full_like(x, np.nan)
        prev[w:] = x[:-w]
        return x - prev
    elif op == 'zscore':
        m = df.rolling(w, min_periods=max(3, w // 2)).mean()
        s = df.rolling(w, min_periods=max(3, w // 2)).std()
        r = (df - m) / s.replace(0, np.nan)
        return r.values.astype(np.float32)
    else:
        raise ValueError(op)
    return r.values.astype(np.float32)


def _ts_binary(op, a, b, w, ctx):
    if np.ndim(a) == 0:
        a = np.full(b.shape, a, dtype=np.float32)
    if np.ndim(b) == 0:
        b = np.full(a.shape, b, dtype=np.float32)
    da, db = ctx.df(a), ctx.df(b)
    mp = min(w, max(3, w // 2))          # min_periods 不能超过窗口长度
    if op == 'corr':
        r = da.rolling(w, min_periods=mp).corr(db)
    elif op == 'cov':
        r = da.rolling(w, min_periods=mp).cov(db)
    else:
        raise ValueError(op)
    return r.values.astype(np.float32)


def _cs(op, x, ctx):
    if np.ndim(x) == 0:
        return np.full(ctx.p.pstar.shape, x, dtype=np.float32)
    df = ctx.df(x)
    if op == 'rank':
        return df.rank(axis=1, pct=True).values.astype(np.float32)
    if op == 'zscore':
        m = df.mean(axis=1)
        s = df.std(axis=1).replace(0, np.nan)
        return df.sub(m, axis=0).div(s, axis=0).values.astype(np.float32)
    if op == 'demean':
        return df.sub(df.mean(axis=1), axis=0).values.astype(np.float32)
    raise ValueError(op)


def sanitize(x):
    """把 inf / 极端值压掉, 保持横截面可比."""
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    return x
