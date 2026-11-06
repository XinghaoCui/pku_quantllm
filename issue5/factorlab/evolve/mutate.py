"""随机生成 / 变异 / 交叉."""
from __future__ import annotations

import random

from . import expr


def _rand_var(rng):
    return ('v', rng.choice(list(expr.VARS)))


def _rand_leaf(rng):
    if rng.random() < 0.12:
        return ('c', rng.choice([0.5, 1.0, 2.0, -1.0]))
    return _rand_var(rng)


def random_tree(rng: random.Random, max_nodes=6) -> tuple:
    """生成一棵不超过 max_nodes 的随机表达式."""
    def grow(budget):
        if budget <= 1:
            return _rand_leaf(rng)
        r = rng.random()
        if r < 0.30:
            return _rand_leaf(rng)
        if r < 0.48:
            op = rng.choice(list(expr.UNARY))
            return ('u', op, grow(budget - 1))
        if r < 0.62:
            op = rng.choice(list(expr.CS))
            return ('cs', op, grow(budget - 1))
        if r < 0.86:
            op = rng.choice([o for o in expr.TS_UNARY if o != 'delay'])
            return ('t', op, grow(budget - 1), rng.choice(list(expr.WINDOWS)))
        if r < 0.94 and budget >= 3:
            op = rng.choice(list(expr.TS_BINARY))
            return ('t2', op, grow(budget - 2), grow(budget - 2),
                    rng.choice(list(expr.WINDOWS)))
        op = rng.choice(list(expr.BINARY))
        half = (budget - 1) // 2
        return ('b', op, grow(half), grow(budget - 1 - half))

    n = rng.randint(2, max_nodes)
    for _ in range(20):
        t = grow(n)
        if 2 <= expr.nodes(t) <= max_nodes + 2:
            return t
    return ('t', 'pct', ('v', 'pstar'), rng.choice(list(expr.WINDOWS)))


def _paths(n, acc=None, path=()):
    acc = [] if acc is None else acc
    acc.append(path)
    k = n[0]
    if k in ('u', 'cs'):
        _paths(n[2], acc, path + (0,))
    elif k == 'b':
        _paths(n[2], acc, path + (0,)); _paths(n[3], acc, path + (1,))
    elif k == 't':
        _paths(n[2], acc, path + (0,))
    elif k == 't2':
        _paths(n[2], acc, path + (0,)); _paths(n[3], acc, path + (1,))
    return acc


def get_at(n, path):
    for i in path:
        k = n[0]
        if k in ('u', 'cs', 't'):
            n = n[2]
        elif k in ('b', 't2'):
            n = n[2] if i == 0 else n[3]
        else:
            raise ValueError(path)
    return n


def replace_at(n, path, new):
    if not path:
        return new
    i, rest = path[0], path[1:]
    k = n[0]
    if k in ('u', 'cs', 't'):
        return (k, n[1], replace_at(n[2], rest, new)) + tuple(n[3:])
    if k in ('b', 't2'):
        a = replace_at(n[2], rest, new) if i == 0 else n[2]
        b = replace_at(n[3], rest, new) if i == 1 else n[3]
        return (k, n[1], a, b) + tuple(n[4:])
    raise ValueError(path)


def mutate(rng, node, max_nodes=8):
    """五类变异: 换窗口 / 换算子 / 包裹 / 换子树 / 交换操作数."""
    ps = _paths(node)
    kind = rng.choice(['window', 'op', 'wrap', 'subtree', 'swap'])
    try:
        if kind == 'window':
            ws = [p for p in ps if get_at(node, p)[0] in ('t', 't2')]
            if not ws:
                return mutate(rng, node, max_nodes)
            p = rng.choice(ws)
            sub = get_at(node, p)
            w = rng.choice([x for x in expr.WINDOWS if x != sub[3 if sub[0] == 't' else 4]])
            if sub[0] == 't':
                return replace_at(node, p, (sub[0], sub[1], sub[2], w))
            return replace_at(node, p, (sub[0], sub[1], sub[2], sub[3], w))
        if kind == 'op':
            cands = [p for p in ps if get_at(node, p)[0] in ('u', 'b', 'cs', 't')]
            if not cands:
                return random_tree(rng, max_nodes)
            p = rng.choice(cands)
            sub = get_at(node, p)
            if sub[0] == 'u':
                return replace_at(node, p, ('u', rng.choice(list(expr.UNARY)), sub[2]))
            if sub[0] == 'b':
                return replace_at(node, p, ('b', rng.choice(list(expr.BINARY)), sub[2], sub[3]))
            if sub[0] == 'cs':
                return replace_at(node, p, ('cs', rng.choice(list(expr.CS)), sub[2]))
            return replace_at(node, p, ('t', rng.choice([o for o in expr.TS_UNARY if o != 'delay']),
                                        sub[2], sub[3]))
        if kind == 'wrap':
            p = rng.choice(ps)
            sub = get_at(node, p)
            r = rng.random()
            if r < 0.45:
                new = ('u', rng.choice(['neg', 'abs', 'log', 'sqrt_abs']), sub)
            elif r < 0.75:
                new = ('t', rng.choice(['mean', 'std', 'zscore']), sub, rng.choice(list(expr.WINDOWS)))
            else:
                new = ('cs', rng.choice(list(expr.CS)), sub)
            if expr.nodes(new) > max_nodes + 2:
                return node
            return replace_at(node, p, new)
        if kind == 'subtree':
            p = rng.choice(ps)
            new = random_tree(rng, max(2, max_nodes - expr.depth(get_at(node, p))))
            if expr.nodes(replace_at(node, p, new)) > max_nodes + 3:
                return node
            return replace_at(node, p, new)
        # swap
        bs = [p for p in ps if get_at(node, p)[0] == 'b']
        if not bs:
            return random_tree(rng, max_nodes)
        p = rng.choice(bs)
        sub = get_at(node, p)
        return replace_at(node, p, ('b', sub[1], sub[3], sub[2]))
    except (ValueError, IndexError):
        return node


def crossover(rng, a, b):
    pa = _paths(a)
    pb = _paths(b)
    if not pa or not pb:
        return a
    x = get_at(a, rng.choice(pa))
    y = get_at(b, rng.choice(pb))
    return replace_at(a, rng.choice(pa), y)
