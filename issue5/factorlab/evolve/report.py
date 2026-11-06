"""把演化结果写成 JSON / CSV / Markdown."""
from __future__ import annotations

import json
import math

import numpy as np

from . import expr


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def pool_json(pool):
    return [{'cid': c.cid, 'formula': expr.to_str(c.node), 'nodes': expr.nodes(c.node),
             'rank_ic': _clean(c.stats['rank_ic']), 't': _clean(c.stats['t']),
             'icir': _clean(c.stats['icir']), 'corr': _clean(c.corr),
             'incr': _clean(c.incr), 'mono': _clean(c.stats['mono'])}
            for c in sorted(pool, key=lambda c: -c.ic)]


def write_json(path, meta, arms, random_res, bias, learning):
    out = {
        'meta': _clean(meta),
        'arms': [_clean(a) for a in arms],
        'random': _clean(random_res),
        'selection_bias': _clean(bias),
        'learning': _clean(learning),
    }
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              separators=(',', ':'), allow_nan=False)
    print(f'  evolution.json {len(json.dumps(out))/1024:.0f} KB -> {path}')


def write_markdown(path, meta, arms, random_res, bias, learning):
    L = []
    a = L.append
    a('# 第六讲 · 因子演化实验\n')
    a(f'> 自动生成 · {meta["generated"]}\n')
    a('## 1. 实验设计\n')
    a(f'- 训练区间 **{meta["train"][0]} ~ {meta["train"][1]}**（{meta["n_train_days"]} 个交易日），'
      f'验证 2024，样本外 **{meta["oos"][0]} ~ {meta["oos"][1]}** 只在冻结后跑一次。')
    a(f'- 固定预算：每组共享 {len(meta["seeds"])} 个初始因子，再做 **{meta["budget"]} 次**新增候选的完整评价；'
      '完整评价 = 计算因子矩阵 + 训练期 RankIC/HAC/分组 + 与因子池的相关与增量。')
    a('- 标签与评价口径沿用第五讲：t 收盘信号 → t+1 开盘进场 → t+1+5 开盘出场，复权价。')
    a('')
    a('| 组别 | 允许改变的内容 |')
    a('|---|---|')
    a('| 基线 | 固定流程：只保留一个当前最优，子代均匀变异，`\\|RankIC\\|` 更高就替换 |')
    a('| Harness | 因子池、多样性、相关/增量门控、按接受率调整变异算子 |')
    a('| 学习 | 在 Harness 上增加候选评分器：先按结构特征预测质量，只对预测更好的候选做完整评价 |')
    a('| 随机 | 纯随机公式，不迭代，用于估计"搜索次数本身"带来的选择偏差 |')
    a('')
    a('## 2. 三组对照\n')
    a('| 组别 | 完整评价 | 接受 | 池内最好 \\|RankIC\\| | 池均值 \\|RankIC\\| | 样本外 RankIC | 样本外净收益 | 同池等权 | 超额 |')
    a('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for x in arms:
        o = x.get('oos') or {}
        a(f'| {x["name"]} | {x["evals"]} | {x["accepts"]} | '
          f'{max([p["rank_ic"] for p in x["pool"]], key=abs, default=float("nan")):+.4f} | '
          f'{np.mean([abs(p["rank_ic"]) for p in x["pool"]]):.4f} | '
          f'{o.get("rank_ic", float("nan")):+.4f} | '
          f'{o.get("net_return", float("nan"))*100:+.2f}% | '
          f'{o.get("bench_return", float("nan"))*100:+.2f}% | '
          f'{o.get("excess", float("nan"))*100:+.2f}pp |')
    a('')
    a('> 样本外净收益是同分数前 100 只等权、每 5 日调仓、扣佣金/印花税/滑点后的账户结果；'
      '同池等权是零成本的分散化参照，两者分散度不同，超额不能全读作因子的 alpha。\n')
    a('## 3. 随机对照与选择偏差\n')
    r = random_res
    a(f'- 随机生成 {r["n"]} 个公式：训练期 \\|RankIC\\| 中位数 {r["median_ic"]:.4f}，'
      f'最大值 {r["max_ic"]:.4f}（{r["best"]["formula"]}）。')
    a(f'- 同一个"最佳随机公式"（训练期 {r["best_train_vs_oos"]:+.4f}）按训练期方向对齐后，'
      f'样本外 RankIC 为 {r["best_oos_ic"]:+.4f}。')
    a(f'- 零假设下 N 次搜索的最大 \\|RankIC\\| 期望约为 `sqrt(2·ln N / T)`：'
      f'N={bias["n"][-1]}、T={meta["n_train_days"]} 时为 {bias["expected_max"][-1]:.4f}；'
      f'随机组实际最大 {bias["observed_max"][-1]:.4f}。')
    a('')
    a('这条曲线不是"显著性阈值"，它说明：**候选越多，历史上的最好成绩越好看**，'
      '即使所有候选真实均值都是 0。所以下面所有结论都以冻结后的样本外为准。\n')
    a('## 4. 学习型候选评分器\n')
    if learning.get('n_train', 0) >= 15:
        a(f'- 训练样本 {learning["n_train"]} 个（Learning 组已完整评价过的候选，含被模型过滤前的历史）。')
        a(f'- 在时间上靠后的 40% 候选上，预测值与实际 \\|RankIC\\| 的 Spearman 相关为 '
          f'**{learning["spearman_pred_actual"]:+.3f}**。')
        a('- 特征只有结构信息（节点数、深度、窗口、是否含横截面算子、父代分数），'
          '不包含候选自身的评价结果；因此它可以用来省评价预算，但不构成新的 alpha 证据。')
    else:
        a('- 本轮样本不足以拟合评分器，Learning 组退化为 Harness 组。')
    a('')
    a('## 5. 结论\n')
    a(_conclusion(arms, random_res))
    a('')
    a('## 6. 局限\n')
    a('- 演化在训练区间内做选择，训练区间此前已被反复研究；样本外只有 424 个交易日，'
      '且 2025–2026 的行情也已在其他作业中出现，**不是真正的盲测**。')
    a('- 表达式空间只有一小组因果算子，因子值按日横截面排序，未做行业/市值中性化。')
    a('- 完整评价用训练期 IC，未把换手与成本放进适应度；账户检验只在最后做一次。')
    a('- 学习型评分器只预测"训练期 |RankIC|"，它优化的是搜索效率，不是样本外表现。')
    a('')
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'  EVOLUTION.md -> {path}')


def _conclusion(arms, random_res):
    best = max(arms, key=lambda a: (a.get('oos') or {}).get('rank_ic', -9))
    o = best.get('oos') or {}
    lines = [f'- 样本外 RankIC 最高的是 **{best["name"]}**（{o.get("rank_ic", float("nan")):+.4f}），'
             f'扣费净收益 {o.get("net_return", float("nan"))*100:+.2f}%，'
             f'相对同池等权 {o.get("excess", float("nan"))*100:+.2f}pp。']
    rnd = random_res['best_train_vs_oos']
    lines.append(f'- 随机对照里训练期 \|RankIC\| 最高的公式（{rnd:+.4f}），样本外按同方向只剩 '
                 f'{random_res["best_oos_ic"]:+.4f}；再结合零假设曲线，'
                 f'训练期的"高分"里有相当一部分来自搜索次数本身。')
    lines.append('- 这次没有找到能在样本外稳定跑赢同池等权的因子组合；把负结果和全部候选都保留下来，'
                 '比再调几轮参数更接近这门课强调的"可复现实验"。')
    return '\n'.join(lines)
