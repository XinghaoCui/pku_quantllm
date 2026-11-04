"""把评价结果压成一份网页可直接用的 JSON."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def _deepclean(o):
    if isinstance(o, dict):
        return {k: _deepclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_deepclean(v) for v in o]
    return _clean(o)


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def write_dashboard(p, summary, sp_df, corr_df, resid_df, trad, ic_rows, args, path):
    dates = p.dates
    ic = {}
    if ic_rows:
        allic = pd.concat(ic_rows)
        for k, g in allic.groupby('factor'):
            ic[k] = {'dates': [d.strftime('%Y-%m-%d') for d in g['date']],
                     'rank_ic': [round(float(v), 5) for v in g['rank_ic']]}

    # 累计 RankIC 曲线 (求和, 便于看趋势) + 63 日滚动均值
    for k, v in ic.items():
        s = pd.Series(v['rank_ic'])
        v['cum'] = [round(float(x), 5) for x in s.cumsum()]
        v['ma63'] = [None if pd.isna(x) else round(float(x), 5)
                     for x in s.rolling(63, min_periods=20).mean()]

    resid = {r['factor']: {k: _clean(v) for k, v in r.items() if k != 'factor'}
             for r in resid_df.to_dict('records')} if len(resid_df) else {}

    trad_rows = []
    if len(trad):
        for r in trad.to_dict('records'):
            trad_rows.append({
                'factor': r['factor'], 'name': r['name'],
                'cum_return': _clean(r['cum_return']), 'ann_return': _clean(r['ann_return']),
                'sharpe': _clean(r['sharpe']), 'mdd': _clean(r['mdd']),
                'bench_return': _clean(r['bench_return']), 'excess': _clean(r['excess']),
                'turnover_ann': _clean(r['turnover_ann']), 'cost': _clean(r['cost']),
                'residual_value': _clean(r['residual_value']), 'n_trades': _clean(r['n_trades']),
            })

    curves = {}
    try:
        curves = json.load(open(path.replace('dashboard.json', 'tradable_curves.json')))
    except Exception:
        pass

    out = {
        'meta': {
            'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'period': [dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')],
            'n_codes': int(p.close.shape[1]), 'n_days': int(p.close.shape[0]),
            'st': p.st_stats,
            'eval_start': args.start, 'eval_end': args.end,
            'horizons': [1, 5, 10, 20],
            'splits': {'train': ['2021-01-01', '2023-12-31'],
                       'valid': ['2024-01-01', '2024-12-31'],
                       'test': ['2025-01-01', '2026-09-30']},
            'label': 't 收盘信号 -> t+1 开盘进场 -> t+1+h 开盘出场, 复权',
            'costs': '佣金万一(单边,最低5元) + 卖出印花税万五 + 单边5bp滑点',
        },
        'factors': [{'key': e['factor'], 'name': e['name'], 'category': e['category'],
                     'formula': e['formula'], 'direction': e['direction'],
                     'verdict': e['verdict'], 'checks': e['checks'],
                     'train': e['train_rank_ic'] * e['direction'],
                     'valid': e['valid_rank_ic'] * e['direction'],
                     'test': e['test_rank_ic'], 'test_signed': e['test_rank_ic_signed'],
                     'test_t': e['test_t'], 'ic5': e['rank_ic_5'], 't5': e['t_5'],
                     'icir5': e['icir_5'] * e['direction'], 'pos5': e['pos5_signed'],
                     'mono5': e['mono5_signed'],
                     'ls5': e['ls_5'] * e['direction'],
                     'ls_sharpe5': e['ls_sharpe_5'] * e['direction'],
                     'fdr5': e['fdr_5'], 'decay': e['decay'],
                     'quantiles': e['quantiles'], 'avg_names': e['avg_names']}
                    for e in summary],
        'splits': _deepclean(sp_df.round(6).to_dict('records')),
        'ic': ic,
        'corr': {'keys': list(corr_df.index),
                 'matrix': [[_clean(v) for v in row] for row in corr_df.values]},
        'residual': resid,
        'tradable': trad_rows,
        'curves': curves,
    }
    out = _deepclean(out)
    json.dump(out, open(path, 'w'), ensure_ascii=False, separators=(',', ':'),
              allow_nan=False)
    print(f'  dashboard.json {len(json.dumps(out))/1024:.0f} KB -> {path}')


def write_markdown(summary, sp_df, resid_df, trad, p, path):
    """生成 REPORT.md: 只用已算好的结果, 不重新读行情."""
    import os
    rows = sorted(summary, key=lambda e: -(e['test_rank_ic_signed'] or -9))
    tm = {r['factor']: r for r in trad.to_dict('records')} if len(trad) else {}
    rd = {r['factor']: r for r in resid_df.to_dict('records')} if len(resid_df) else {}
    L = []
    a = L.append
    a('# 第五讲 · 因子与因子评价\n')
    a(f'> 由 `run_lab.py` 自动生成 · {pd.Timestamp.now():%Y-%m-%d %H:%M}\n')
    a('## 1. 结论摘要\n')
    best = rows[:3]
    a(f'- 评价 {len(summary)} 个内置因子 × 4 个预测周期，'
      f'样本 {p.dates[0]:%Y-%m-%d} ~ {p.dates[-1]:%Y-%m-%d}，'
      f'日度股票池中位 {int(np.median([e["avg_names"] for e in summary]))} 只。')
    a('- 5 日 RankIC 最高：' + '、'.join(
        f'`{e["factor"]}` {e["rank_ic_5"]:+.4f}（HAC t={e["t_5"]:+.2f}）' for e in
        sorted(summary, key=lambda x: -abs(x['rank_ic_5']))[:3]) + '。')
    if tm:
        win = [r['factor'] for r in tm.values() if r['excess'] > 0]
        if win:
            a(f'- **预测力不等于可交易性**：{len(tm)} 个因子里只有 {len(win)} 个（'
              + '、'.join(f'`{k}`' for k in win) + '）的扣费多头账户跑赢同池等权基准。')
        else:
            a('- **预测力不等于可交易性**：全部因子的扣费多头账户都跑输同池等权基准。')
    if rd:
        gone = [k for k, v in rd.items() if abs(v['resid_rankin']) < 0.01 and abs(v['raw_rankin']) > 0.02]
        keep = [k for k, v in rd.items() if v['resid_rankin'] >= 0.02]
        a(f'- 控制 log 价格／log 成交额／20 日波动／20 日动量后，'
          f'{len(gone)} 个因子的 IC 基本归零（{"、".join(f"`{k}`" for k in gone)}），'
          f'说明它们的预测力主要来自规模/流动性/波动暴露。')
        if keep:
            a(f'- 残差 IC 仍 ≥0.02 的：{"、".join(f"`{k}`" for k in keep)}。')
    a('')
    a('## 2. 数据与股票池\n')
    st = p.st_stats
    a(f'- 课程 `feathers.zip` 七张宽表：{p.close.shape[0]} 个交易日 × {p.close.shape[1]} 只证券。')
    a(f'- ST：BaoStock 逐日 `isST` 覆盖 {st["official"]} 只（2024 年起），'
      f'其余日期/证券用 ±5% 涨跌幅行为推断填 {st["proxy_filled_cells"]:,} 格；未知状态一律不买。')
    a('- 每日股票池：非 ST、有收盘价、上市满 252 个交易日、20 日日均成交额 ≥ 2000 万元。')
    a('')
    a('## 3. 评价口径\n')
    a('| 环节 | 做法 |')
    a('|---|---|')
    a('| 标签 | t 日收盘出信号 → t+1 开盘进场 → t+1+h 开盘出场，`O*(t+1+h)/O*(t+1)-1`，复权价，不用收盘对收盘 |')
    a('| 切分 | 训练 2021–2023 / 验证 2024 / 测试 2025-01–2026-09，跨段标签剔除 |')
    a('| 方向 | 只用训练段 RankIC 的符号冻结，验证/测试不翻转 |')
    a('| 显著性 | Newey-West(HAC, lag=h) t 值 + Benjamini-Hochberg FDR（全因子×周期为检验族）|')
    a('| 分组 | 每日按因子值分 5 组，报各组平均未来收益、单调性、多空 Sharpe |')
    a('| 中性化 | 对 log 价格、log ADV20、20 日波动、20 日动量的横截面秩做 OLS，取残差 IC |')
    a('| 交易 | 头组等权 100 只、每 5 日调仓、10 亿、佣金万一/卖出印花税万五/单边 5bp 滑点、T+1、整手、涨跌停拒单、容量 5% |')
    a('')
    a('## 4. 主结果\n')
    a('按测试段（方向冻结后）RankIC 排序；训练/验证/测试三列都已乘上冻结方向。判定门槛：测试 RankIC≥0.02、HAC t≥2、正 IC 比例≥0.55、'
      '分组单调性≥0.6、扣费净超额>0，五项全过为「好因子」，≥3 项且同向为「有潜力」。\n')
    a('| 因子 | 公式 | 训练 | 验证 | 测试 | 测试 t | ICIR | 正IC | 单调 | 判定 | 净超额 |')
    a('|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|')
    for e in rows:
        t = tm.get(e['factor'])
        ex = f'{t["excess"]*100:+.1f}pp' if t else '—'
        a(f'| `{e["factor"]}` {e["name"]} | `{e["formula"]}` | {e["train_rank_ic"]*e["direction"]:+.4f} | '
          f'{e["valid_rank_ic"]*e["direction"]:+.4f} | {e["test_rank_ic_signed"]:+.4f} | {e["test_t_signed"]:+.2f} | '
          f'{e["icir_5"]*e["direction"]:+.3f} | {e["pos5_signed"]*100:.1f}% | {e["mono5_signed"]:+.2f} | '
          f'{e["verdict"]} | {ex} |')
    a('')
    a('> 测试段列已乘上训练期冻结的方向；`mom20` 与 `rev20` 互为相反数，方向冻结后持仓完全相同。\n')
    a('## 5. 三张表要一起看\n')
    a('1. **预测力**（IC/t/分组单调）：衡量分数与未来收益的横截面关系；')
    a('2. **稳定性**（训练→验证→测试）：IC 衰减或反号，说明关系随市场状态变化；')
    a('3. **可交易性**（扣费账户）：换手决定了成本，100 只集中持仓又放大了容量与个股风险。')
    a('')
    a('三者可以完全背离：本次 5 日反转 `rev5` 的 RankIC 显著为正，但每 5 日几乎换掉全部持仓，'
      '双边换手极高，扣费后跑输基准 60 个百分点以上。')
    a('')
    a('## 6. 局限\n')
    a('- 日线撮合：开盘价成交是近似，无法还原排队与盘中路径；涨跌停只用"开盘是否封板"拒单。')
    a('- ST 在 2024 年之前依赖价格行为推断（precision≈0.88 / recall≈0.39，见 `build_st_proxy.py`），'
      '可能漏掉部分 ST，也可能误杀少量非 ST。')
    a('- 中性化只有价格、成交额、波动、动量四个代理，缺少行业与真实市值；'
      '"残差 IC 仍为正"不等于独立 alpha。')
    a('- 期末卖不掉的残仓按收盘价计入净值，不能当作已变现现金。')
    a('- 基准是同池 4000+ 只等权、零成本、同频率调仓；策略是 100 只集中组合。'
      '超额里混着分散度、规模与风格效应，不能全部读作因子的 alpha。')
    a('- 2021–2026 的历史已在研究过程中被反复查看，测试段不是真正的盲测。')
    a('')
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'  REPORT.md -> {path}')
