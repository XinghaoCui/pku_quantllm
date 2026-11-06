"""第六讲 · 因子演化实验.

固定预算下比较四组:
  基线 / Harness / 学习型候选评分器 / 纯随机对照

用法:
    py run_evolution.py --data-dir ../feathers --st ../st_mask.parquet \
                        --st-proxy ../st_proxy.parquet --budget 64 --rounds 8
    py run_evolution.py --budget 16 --rounds 4 --random 16      # 快速试跑
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import log as _log, sqrt

import numpy as np
import pandas as pd

from factorlab import evaluate as E
from factorlab import tradable as T
from factorlab.data import Panel
from factorlab.evolve import expr, harness, report
from factorlab.evolve.harness import SEEDS, Evolution, random_candidates


def log(*a):
    print(*a, flush=True)


def build_mask(p, days):
    masks = np.zeros(p.close.shape, dtype=bool)
    for t in days:
        masks[t] = p.universe(t)
    return masks


def oos_pool(p, ctx, pool, oos_days, masks, label, start, end):
    """冻结后的池: 等权秩平均合成分数, 算样本外 RankIC 与扣费账户."""
    comp = np.zeros(p.close.shape, dtype=np.float64)
    cnt = 0
    per_member = []
    for c in pool:
        fac = expr.sanitize(ctx.evaluate(c.node))
        df = pd.DataFrame(fac, index=p.dates)
        z = df.rank(axis=1, pct=True).values
        d = 1.0 if c.stats['rank_ic'] >= 0 else -1.0
        comp += np.nan_to_num(z) * d
        cnt += 1
        st = E.factor_stats(fac, label, masks, oos_days, 5)
        per_member.append({'cid': c.cid, 'formula': expr.to_str(c.node),
                           'oos_rank_ic': st['rank_ic_mean'], 'oos_t': st['t_hac']})
    comp = (comp / max(cnt, 1)).astype(np.float32)
    comp[~np.isfinite(p.pstar)] = np.nan
    st = E.factor_stats(comp, label, masks, oos_days, 5)
    tr = T.run_factor(p, comp, 1.0, start, end, top=100, rebal=5)
    return {
        'rank_ic': st['rank_ic_mean'], 't': st['t_hac'], 'n': st['n_days'],
        'net_return': tr['cum_return'], 'bench_return': tr['bench_return'],
        'excess': tr['excess'], 'mdd': tr['mdd'], 'sharpe': tr['sharpe'],
        'turnover': tr['turnover_ann'], 'cost': tr['cost'],
        'residual_value': tr['residual_value'],
        'equity': {'dates': [d.strftime('%Y-%m-%d') for d in tr['equity']['date']],
                   'nav': [round(float(v), 5) for v in tr['equity']['nav']],
                   'bench': [round(float(v), 5) for v in tr['equity']['bench']]},
        'members': per_member,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../feathers')
    ap.add_argument('--st', default='../st_mask.parquet')
    ap.add_argument('--st-proxy', default='../st_proxy.parquet')
    ap.add_argument('--out', default='results/evolution')
    ap.add_argument('--budget', type=int, default=64)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--random', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--start', default='2025-01-02')
    ap.add_argument('--end', default='2026-09-30')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    p = Panel(args.data_dir, args.st, args.st_proxy)
    log(f'panel {p.close.shape}, ST {p.st_stats}')

    nD = len(p.dates)
    label = E.fwd_labels(p, 5)
    train = np.arange(p.idx('2021-01-01'), p.last_le('2023-12-31') + 1)
    sample = train[::5]
    oos = np.arange(p.idx(args.start), min(p.last_le(args.end), nD - 6) + 1)
    masks = build_mask(p, np.arange(E.WARMUP_DAYS - 5, nD - 6))
    log(f'train {len(train)} days, sample {len(sample)}, oos {len(oos)} days')

    ctx = expr.Ctx(p)
    evo = Evolution(ctx, train, sample, label, masks,
                    budget=args.budget, rounds=args.rounds, seed=args.seed)

    arms_out = []
    all_cands = []
    runs = []
    for arm, name, desc in [
        ('baseline', '基线', '固定流程：单点最优 + 均匀变异'),
        ('harness', 'Harness', '因子池 + 多样性 + 门控 + 算子反馈'),
        ('learning', '学习', 'Harness + 结构特征候选评分器'),
    ]:
        t1 = time.time()
        res = evo.run(arm)
        runs.append(res)
        log(f'\n[{name}] evals={res["evals"]} accepts_new={res["accepts_new"]} '
            f'({time.time()-t1:.0f}s)')
        oos_res = oos_pool(p, ctx, res['pool'], oos, masks, label, args.start, args.end)
        log(f'  pool ic ' + ', '.join(f'{c.stats["rank_ic"]:+.4f}' for c in res['pool']))
        log(f'  OOS  ic {oos_res["rank_ic"]:+.4f}  net {oos_res["net_return"]:+.2%}  '
            f'bench {oos_res["bench_return"]:+.2%}')
        all_cands += [c.row() for c in res['cands']]
        arms_out.append({
            'key': arm, 'name': name, 'desc': desc,
            'evals': res['evals'], 'accepts': res['accepts_new'],
            'pool': report.pool_json(res['pool']),
            'trace': res['trace'],
            'oos': {k: v for k, v in oos_res.items() if k != 'equity'},
            'equity': oos_res['equity'],
        })

    # 随机对照
    t1 = time.time()
    rnd = random_candidates(evo, args.random, seed=args.seed + 1)
    rnd_rows = [c.row() for c in rnd]
    all_cands += rnd_rows
    best = max(rnd, key=lambda c: c.ic)
    rnd.sort(key=lambda c: -c.ic)
    best_oos = E.factor_stats(expr.sanitize(ctx.evaluate(best.node)), label, masks, oos, 5)
    log(f'\n[随机] n={len(rnd)} median |IC| {np.median([c.ic for c in rnd]):.4f} '
        f'max {best.ic:.4f} -> OOS {best_oos["rank_ic_mean"]:+.4f} '
        f'({time.time()-t1:.0f}s)')

    sgn = 1.0 if best.stats['rank_ic'] >= 0 else -1.0
    random_res = {
        'n': len(rnd),
        'median_ic': float(np.median([c.ic for c in rnd])),
        'max_ic': float(best.ic),
        'best': {'formula': expr.to_str(best.node), 'rank_ic': best.stats['rank_ic'],
                 't': best.stats['t']},
        'best_train_vs_oos': best.stats['rank_ic'],
        'best_oos_raw': best_oos['rank_ic_mean'],
        'best_oos_ic': best_oos['rank_ic_mean'] * sgn,      # 按训练期方向对齐
        'best_dir': sgn,
        'items': [{'formula': r['formula'], 'rank_ic': r['rank_ic'], 't': r['t']}
                  for r in rnd_rows[:40]],
    }

    # 选择偏差: 零假设下 N 次搜索的最大 |RankIC|
    T = len(train)
    ns = [8, 16, 32, 64, 128]
    expected = [float(sqrt(2 * _log(n) / T)) for n in ns]
    # 经验分布: 从全部随机候选里无放回抽 n 个, 重复 300 次, 取"最大值"的均值
    rng_b = np.random.default_rng(0)
    ics = np.array([c.ic for c in rnd])
    observed = []
    for n in ns:
        k = min(n, len(ics))
        if k == 0:
            observed.append(0.0); continue
        draws = [ics[rng_b.choice(len(ics), k, replace=False)].max() for _ in range(300)]             if k < len(ics) else [ics.max()]
        observed.append(float(np.mean(draws)))
    bias = {'n': ns, 'expected_max': expected, 'observed_max': observed}

    # 学习型评分器质量: 时间靠后的 40% 候选上, 预测 vs 实际
    lc = [c for c in evo.cands if c.arm == 'learning' and c.stats and np.isfinite(c.pred)]
    learning = {'n_train': len(lc), 'spearman_pred_actual': None,
                'pred': [], 'actual': []}
    if len(lc) >= 20:
        cut = int(len(lc) * 0.6)
        test = lc[cut:] or lc
        pred = np.array([c.pred for c in test])
        act = np.array([c.ic for c in test])
        from scipy.stats import spearmanr
        learning['spearman_pred_actual'] = float(spearmanr(pred, act).statistic)
        learning['n_train'] = len(lc)
        learning['pred'] = [round(float(x), 4) for x in pred]
        learning['actual'] = [round(float(x), 4) for x in act]
        log(f'\n[学习] 评分器在 {len(test)} 个候选上 pred~actual Spearman '
            f'{learning["spearman_pred_actual"]:+.3f}')

    meta = {
        'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'train': ['2021-01-01', '2023-12-31'],
        'valid': ['2024-01-01', '2024-12-31'],
        'oos': [args.start, args.end],
        'n_codes': int(p.close.shape[1]), 'n_train_days': int(T),
        'budget': args.budget, 'rounds': args.rounds,
        'seeds': runs[1]['seed_stats'] if len(runs) > 1 else
                 [{'name': n, 'formula': expr.to_str(x)} for n, x in SEEDS],
        'label': 't 收盘信号 -> t+1 开盘进场 -> t+6 开盘出场, 复权',
        'costs': '佣金万一(最低5元) + 卖出印花税万五 + 单边5bp滑点',
        'st': p.st_stats,
    }

    report.write_json(f'{args.out}/evolution.json', meta, arms_out, random_res,
                      bias, learning)
    report.write_markdown(f'{args.out}/EVOLUTION.md', meta, arms_out, random_res,
                          bias, learning)

    pd.DataFrame(all_cands).to_csv(f'{args.out}/candidates.csv', index=False)
    for a in arms_out:
        pd.DataFrame(a['trace']).to_csv(f'{args.out}/trace_{a["key"]}.csv', index=False)
    lineage = []
    for a in arms_out:
        res = next(x for x in runs if x['arm'] == a['key'])
        for c in res['cands']:
            if c.accepted and c.reason not in ('seed', 'seed_unused', 'replaced'):
                lineage.append({'arm': a['key'], 'cid': c.cid, 'parent': c.parent or '',
                                'round': c.rnd, 'formula': expr.to_str(c.node),
                                'rank_ic': c.stats['rank_ic'] if c.stats else None,
                                'nodes': expr.nodes(c.node)})
    json.dump(lineage, open(f'{args.out}/lineage.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # 网页数据
    os.makedirs('data', exist_ok=True)
    import shutil
    shutil.copyfile(f'{args.out}/evolution.json', 'data/evolution.json')

    log(f'\ndone in {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
