"""因子研究工作台 · 一键跑完整个评价流程.

用法:
    py run_lab.py --data-dir ../feathers --out results
    py run_lab.py --factors rev5,lowvol20 --tradable 2     # 只跑少数因子 (调试)
    py run_lab.py --no-tradable                            # 跳过账户检验

输出:
    results/factor_summary.csv   每个因子 x 每个预测周期
    results/split_summary.csv    训练/验证/测试分段
    results/ic_series.csv        逐日 IC / RankIC
    results/quantiles.csv        五分组平均未来收益
    results/correlation.csv      因子平均截面相关
    results/residual.csv         代理中性化后的残差 RankIC
    results/tradable.csv         头组多头账户 (含费用)
    results/dashboard.json       网页用的紧凑衍生数据
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from factorlab import evaluate as E
from factorlab import factors as F
from factorlab.data import Panel
from factorlab import tradable as T


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../feathers')
    ap.add_argument('--st', default='../st_mask.parquet')
    ap.add_argument('--st-proxy', default='../st_proxy.parquet')
    ap.add_argument('--out', default='results')
    ap.add_argument('--factors', default='all')
    ap.add_argument('--tradable', default='all', help='all | none | 整数(取前N个)')
    ap.add_argument('--top', type=int, default=100)
    ap.add_argument('--rebal', type=int, default=5)
    ap.add_argument('--start', default='2025-01-02')
    ap.add_argument('--end', default='2026-09-30')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    p = Panel(args.data_dir, args.st, args.st_proxy)
    log(f'panel {p.close.shape}, ST: {p.st_stats}')

    keys = list(F.REGISTRY) if args.factors == 'all' else args.factors.split(',')
    nD = len(p.dates)
    last_eval = nD - 1 - max(E.HORIZONS)
    days = np.arange(E.WARMUP_DAYS, last_eval + 1)
    log(f'eval days {len(days)}: {p.dates[days[0]].date()} ~ {p.dates[days[-1]].date()}')

    # 股票池掩码 (逐日)
    masks = np.zeros((nD, len(p.codes)), dtype=bool)
    for t in days:
        masks[t] = p.universe(t)
    log(f'pool size median {int(np.median(masks[days].sum(axis=1)))}')

    labels = {h: E.fwd_labels(p, h) for h in E.HORIZONS}
    log(f'labels ready {time.time()-t0:.0f}s')

    sample_days = days[::10]
    ranks_by_factor = {}
    summary, splits, ic_rows, q_rows = [], [], [], []

    for k, key in enumerate(keys):
        t1 = time.time()
        fac = F.build(key, p)
        meta = F.REGISTRY[key]
        entry = {'factor': key, 'name': meta[1], 'category': meta[2], 'formula': meta[3]}
        for h in E.HORIZONS:
            st = E.factor_stats(fac, labels[h], masks, days, h)
            entry[f'rank_ic_{h}'] = st['rank_ic_mean']
            entry[f'ic_{h}'] = st['ic_mean']
            entry[f't_{h}'] = st['t_hac']
            entry[f'p_{h}'] = st['p_hac']
            entry[f'icir_{h}'] = st['icir']
            entry[f'pos_{h}'] = st['pos_ratio']
            entry[f'ls_{h}'] = st['ls_mean']
            entry[f'ls_sharpe_{h}'] = st['ls_sharpe']
            entry[f'mono_{h}'] = st['monotonic']
            entry[f'n_{h}'] = st['n_days']
            if h == 5:
                entry['quantiles'] = st['quantiles']
                entry['avg_names'] = st['avg_names']
                ics, rics = st['_series']
                used = p.dates[st['_days']]
                ic_rows.append(pd.DataFrame({
                    'factor': key, 'horizon': h, 'date': used,
                    'rank_ic': rics, 'ic': ics}))
                # 分段
                for sp, (a, b) in E.SPLITS.items():
                    sel = (used >= a) & (used <= b)
                    x = rics[sel]
                    if len(x) > 5:
                        t_s, p_s = E.nw_t(x, lag=5)
                        splits.append({'factor': key, 'split': sp, 'n': int(len(x)),
                                       'rank_ic': float(x.mean()), 't': t_s, 'p': p_s,
                                       'pos': float((x > 0).mean()),
                                       'ir': float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else np.nan})
                    else:
                        splits.append({'factor': key, 'split': sp, 'n': int(len(x)),
                                       'rank_ic': np.nan, 't': np.nan, 'p': np.nan,
                                       'pos': np.nan, 'ir': np.nan})
                q_rows.append({'factor': key, **{f'q{i+1}': v for i, v in enumerate(st['quantiles'])}})
        # 相关性用的秩 (抽样日)
        R = np.full((len(sample_days), len(p.codes)), np.nan, dtype=np.float32)
        for kk, t in enumerate(sample_days):
            m = masks[t] & np.isfinite(fac[t])
            if m.sum() > 50:
                R[kk, m] = pd.Series(fac[t, m]).rank().values
        ranks_by_factor[key] = R
        summary.append(entry)
        log(f'  [{k+1}/{len(keys)}] {key:14s} rankIC5={entry["rank_ic_5"]:+.4f} '
            f't={entry["t_5"]:+.2f} ({time.time()-t1:.1f}s)')
        del fac

    # FDR (以 5 日为主检验, 全体 因子 x 周期)
    pv = [e[f'p_{h}'] for e in summary for h in E.HORIZONS]
    qv = E.bh_fdr(pv)
    it = iter(qv)
    for e in summary:
        for h in E.HORIZONS:
            e[f'fdr_{h}'] = float(next(it))

    # 相关性
    ks = list(ranks_by_factor)
    corr = np.eye(len(ks))
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            A, B = ranks_by_factor[ks[i]], ranks_by_factor[ks[j]]
            cs = []
            for r in range(A.shape[0]):
                ok = np.isfinite(A[r]) & np.isfinite(B[r])
                if ok.sum() > 100:
                    cs.append(E._corr(A[r][ok], B[r][ok]))
            v = float(np.nanmean(cs)) if cs else np.nan
            corr[i, j] = corr[j, i] = v
    corr_df = pd.DataFrame(corr, index=ks, columns=ks)

    # 中性化 (验证+测试段, 5 日)
    resid = []
    sel_days = days[(p.dates[days] >= E.SPLITS['valid'][0])]
    for key in ks:
        fac = F.build(key, p)
        r = E.residual_rank_ic(fac, labels[5], p, masks, sel_days)
        resid.append({'factor': key, **r})
        del fac
    resid_df = pd.DataFrame(resid)

    # 方向冻结 (训练段符号) + 判定
    sp_df = pd.DataFrame(splits)
    for e in summary:
        tr = sp_df[(sp_df.factor == e['factor']) & (sp_df.split == 'train')]
        te = sp_df[(sp_df.factor == e['factor']) & (sp_df.split == 'test')]
        d = 1.0 if (len(tr) == 0 or not np.isfinite(tr.rank_ic.iloc[0]) or tr.rank_ic.iloc[0] >= 0) else -1.0
        e['direction'] = d
        e['train_rank_ic'] = float(tr.rank_ic.iloc[0]) if len(tr) else np.nan
        e['valid_rank_ic'] = float(sp_df[(sp_df.factor == e['factor']) & (sp_df.split == 'valid')].rank_ic.iloc[0])
        e['test_rank_ic'] = float(te.rank_ic.iloc[0]) if len(te) else np.nan
        e['test_rank_ic_signed'] = e['test_rank_ic'] * d
        e['test_t'] = float(te.t.iloc[0]) if len(te) else np.nan
        e['test_t_signed'] = e['test_t'] * d
        e['mono5_signed'] = e['mono_5'] * d
        e['pos5_signed'] = e['pos_5'] if d > 0 else 1.0 - e['pos_5']
        e['test_n'] = int(te.n.iloc[0]) if len(te) else 0
        e['decay'] = [e[f'rank_ic_{h}'] * d for h in E.HORIZONS]

    # 账户检验
    trad = pd.DataFrame()
    if args.tradable != 'none':
        sel = ks if args.tradable == 'all' else ks[:int(args.tradable)]
        rows, curves = [], {}
        for key in sel:
            fac = F.build(key, p)
            d = next(e['direction'] for e in summary if e['factor'] == key)
            st = T.run_factor(p, fac, d, args.start, args.end, top=args.top, rebal=args.rebal)
            rows.append({k: v for k, v in st.items() if not isinstance(v, pd.DataFrame) and k != 'equity'})
            rows[-1]['factor'] = key
            rows[-1]['name'] = next(e['name'] for e in summary if e['factor'] == key)
            eq = st['equity']
            curves[key] = {'dates': [d_.strftime('%Y-%m-%d') for d_ in eq['date']],
                           'nav': [round(float(x), 5) for x in eq['nav']],
                           'bench': [round(float(x), 5) for x in eq['bench']]}
            os.makedirs(f'{args.out}/trades', exist_ok=True)
            st['trades'].to_csv(f'{args.out}/trades/{key}.csv', index=False)
            eq.to_csv(f'{args.out}/trades/{key}_equity.csv', index=False)
            log(f'  trade {key:14s} net {st["cum_return"]:+.2%} bench {st["bench_return"]:+.2%} '
                f'excess {st["excess"]:+.2%} mdd {st["mdd"]:.2%}')
            del fac
        trad = pd.DataFrame(rows)
        trad.to_csv(f'{args.out}/tradable.csv', index=False)
        json.dump(curves, open(f'{args.out}/tradable_curves.json', 'w', encoding='utf-8'))

    # 判定
    tm = {r['factor']: r for r in trad.to_dict('records')} if len(trad) else {}
    for e in summary:
        checks = {
            'test_rank_ic>=0.02': bool(np.isfinite(e['test_rank_ic_signed']) and e['test_rank_ic_signed'] >= 0.02),
            'test_t>=2': bool(np.isfinite(e['test_t_signed']) and e['test_t_signed'] >= 2),
            'test_pos>=0.55': bool(np.isfinite(e['pos5_signed']) and e['pos5_signed'] >= 0.55),
            'mono>=0.6': bool(np.isfinite(e['mono5_signed']) and e['mono5_signed'] >= 0.6),
        }
        t = tm.get(e['factor'])
        checks['net_excess>0'] = bool(t is not None and t['excess'] > 0)
        e['checks'] = checks
        passed = sum(checks.values())
        if e['test_n'] < 126:
            v = '证据不足'
        elif passed == len(checks):
            v = '好因子'
        elif e['test_rank_ic_signed'] > 0 and passed >= 3:
            v = '有潜力'
        elif e['test_rank_ic_signed'] <= 0:
            v = '不合格'
        else:
            v = '证据不足'
        e['verdict'] = v

    # 落盘
    base_cols = ['factor', 'name', 'category', 'formula', 'direction', 'verdict',
                 'train_rank_ic', 'valid_rank_ic', 'test_rank_ic', 'test_rank_ic_signed',
                 'test_t', 'test_n', 'avg_names']
    summ = pd.DataFrame(summary)
    summ[base_cols].to_csv(f'{args.out}/factor_summary.csv', index=False)
    sp_df.to_csv(f'{args.out}/split_summary.csv', index=False)
    pd.concat(ic_rows).to_csv(f'{args.out}/ic_series.csv', index=False)
    pd.DataFrame(q_rows).to_csv(f'{args.out}/quantiles.csv', index=False)
    corr_df.to_csv(f'{args.out}/correlation.csv')
    resid_df.to_csv(f'{args.out}/residual.csv', index=False)

    from factorlab import report as R
    R.write_dashboard(p, summary, sp_df, corr_df, resid_df, trad, ic_rows, args, f'{args.out}/dashboard.json')
    R.write_markdown(summary, sp_df, resid_df, trad, p, f'{args.out}/REPORT.md')
    # 网页读的是 data/dashboard.json (index.html 用相对路径 fetch)
    import shutil
    os.makedirs('data', exist_ok=True)
    shutil.copyfile(f'{args.out}/dashboard.json', 'data/dashboard.json')
    log('  web copy -> data/dashboard.json')
    log(f'done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    sys.exit(main())
