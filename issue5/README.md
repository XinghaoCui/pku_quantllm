# 第五讲 · 因子研究与因子评价

一套可复用的 A 股横截面因子研究工作台：**算因子 → 评估预测力 → 检验稳定性 → 放进带约束的账户里看能不能赚钱**。

- 13 个内置因子 × 4 个预测周期（1/5/10/20 日）
- IC / RankIC / ICIR / 正 IC 比例 / 覆盖率
- Newey-West(HAC) t 值 + Benjamini-Hochberg FDR 多重检验校正
- 五分组平均收益、单调性、多空 Sharpe、IC 衰减
- 训练(2021–2023) / 验证(2024) / 测试(2025-01–2026-09) 三段切分，方向只在训练段冻结
- 控制 log 价格、log 成交额、20 日波动、20 日动量后的残差 IC
- 因子两两截面相关（冗余检查）
- 头组多头账户：T+1、整手、涨跌停拒单、容量上限、佣金/印花税/滑点
- 单文件交互式报告 `index.html`（无需联网、无需构建）
- 27 项单元测试（16 项评价 + 11 项演化）+ 9 项独立复算审计

## 目录

```
issue5/
├── run_lab.py             # 一键跑完整评价流程
├── audit.py               # 独立复算: 因子值/标签/RankIC/现金账本/费用/规则
├── build_st_proxy.py      # 用 ±5% 涨跌幅行为推断 ST, 并在官方标签上验证
├── factorlab/
│   ├── data.py            # 面板加载、ST 合并、涨跌停价、股票池
│   ├── factors.py         # 13 个因子
│   ├── evaluate.py        # IC/分位/切分/中性化/相关性 + HAC + FDR
│   ├── engine.py          # 撮合引擎 (T+1/整手/涨跌停/容量/费用)
│   ├── tradable.py        # 因子 -> 多头账户
│   └── report.py          # dashboard.json + REPORT.md
├── run_evolution.py       # 第六讲: 固定预算下的四组演化对照
├── factorlab/evolve/      # 表达式 DSL / 变异 / 评分 / harness
├── tests/                 # 单元测试 (test_lab.py + test_evolve.py)
├── index.html             # 因子评价交互报告 (需本地 HTTP 服务)
├── evolution.html         # 演化实验交互报告
└── results/               # 运行产出 (CSV / JSON / REPORT.md / trades/ / evolution/)
```

## 运行

### 1. 准备数据

在 `issue5/` 下把课程 `feathers.zip` 解压到 `feathers/`（七个 `.feather`，第一列 `date`）。
原始行情体积大且是课程数据，**不放进仓库**。

ST 历史需要 `issue5/st_mask.parquet`（dates × codes，1=ST）和 `issue5/st_proxy.parquet`。
没有它们也能跑，但所有证券都会被当作"未知 ST"而不买入：

```bash
# 可选: 从 BaoStock 抓逐日 isST (需要联网)
python -c "import baostock" || pip install baostock
python fetch_st.py          # 抓 BaoStock 逐日 isST -> st_mask.parquet
# 可选: 生成价格行为推断的 ST 掩码
python build_st_proxy.py
```

### 2. 跑评价

```bash
cd issue5
python run_lab.py --data-dir ../feathers \
                  --st ../st_mask.parquet --st-proxy ../st_proxy.parquet \
                  --out results
```

约 2 分钟（笔记本 CPU）。调试时只跑两个因子：

```bash
python run_lab.py --factors rev5,lowvol20 --tradable none
```

### 3. 看报告

```bash
python -m http.server 8000
# 浏览器打开 http://localhost:8000/
```

直接双击 `index.html` 会因浏览器限制无法 `fetch` 数据，必须走本地 HTTP。

### 4. 自检

```bash
python -m unittest discover -s tests -v     # 27 项
python audit.py --results results --data-dir feathers \n               --st st_mask.parquet --st-proxy st_proxy.parquet   # 9 项独立复算
```

## 评价口径（摘要）

| 环节 | 做法 |
|---|---|
| 标签 | t 日收盘出信号 → t+1 开盘进场 → t+1+h 开盘出场；`O*(t+1+h)/O*(t+1)-1`，`O*=open×cumadj` |
| 方向 | 只用训练段 RankIC 的符号冻结，验证/测试段不翻转 |
| 显著性 | Newey-West(lag=h) t 值；全因子×周期做 BH-FDR |
| 分组 | 每日按因子值五等分，报各组平均未来收益、单调性、多空 Sharpe |
| 中性化 | 对 log 价格、log ADV20、20 日波动、20 日动量的横截面秩做 OLS 取残差 |
| 交易 | 头组等权 100 只、每 5 日调仓、10 亿、佣金万一(最低5元)+卖出印花税万五+单边 5bp 滑点、T+1、整手、涨跌停拒单、单日单票 ≤ 20 日均额 5% |

判定门槛（五项）：测试段 RankIC ≥ 0.02、HAC t ≥ 2、正 IC 比例 ≥ 0.55、分组单调性 ≥ 0.6、扣费净超额 > 0。
五项全过为「好因子」，≥3 项且同向为「有潜力」，测试段同向 IC ≤ 0 为「不合格」，样本不足 126 日为「证据不足」。
门槛是**本项目自定**，不是老师给的统一标准，换门槛会换标签。

## 结果怎么看

`results/` 里：

- `factor_summary.csv`：每个因子的训练/验证/测试 RankIC、t 值、判定
- `split_summary.csv`：分段明细
- `ic_series.csv`：逐日 IC / RankIC（1371 个交易日）
- `quantiles.csv`：五分组平均未来收益
- `correlation.csv`：因子平均截面相关
- `residual.csv`：原始 vs 残差 RankIC
- `tradable.csv` + `trades/`：扣费账户、逐笔成交、逐日净值
- `data/dashboard.json`：网页读的数据（`run_lab.py` 会自动同步一份）
- `results/dashboard.json`：同上的运行产物
- `REPORT.md`：自动生成的完整报告
- `audit.json`：独立复算结果

**本次运行最重要的三个结论**（数字见 `REPORT.md`）：

1. 低成交额、低成交笔数、低波动的 5 日 RankIC 都在 0.05–0.07，HAC t 值 6–9，很显著。
2. 但把分数变成 100 只的扣费多头账户后，大部分因子跑输同池等权基准；5 日反转最典型——IC 显著为正，换手极高，扣费后大幅跑输。
3. 控制价格/成交额/波动/动量四个代理后，低成交额（+0.066→-0.001）、低波动（+0.065→-0.020）、低成交笔数（+0.053→-0.009）、20 日反转（+0.064→-0.000）的 IC 基本归零，说明它们主要是规模与波动暴露；残差 IC 仍为正且 t 值较大的是「成交额收缩」（+0.034，t=6.7）、「5 日反转」（+0.026，t=4.6）和「日内反转」（+0.025，t=4.8）。

## 第六讲 · 因子演化（同一套程序里的 evolve 模块）

`run_evolution.py` 在**固定评价预算**下比较四组方法（对应课件第 34 页的三组对照 + 一个随机对照）：

| 组别 | 允许改变的内容 |
|---|---|
| 基线 | 只保留一个"当前最优"，子代均匀变异，`\|RankIC\|` 更高就替换 |
| Harness | 8 只因子池 + 多样性 + 相关/增量门控 + 按接受率调整变异算子 |
| 学习 | 在 Harness 上加候选评分器：先用结构特征预测质量，只把完整评价花在预测更好的候选上 |
| 随机 | 纯随机公式、不迭代，用来估计"搜索次数本身"带来的选择偏差 |

每组共享同样 8 个初始因子，再做 **64 次**完整评价；训练 2021–2023，2024 只做确认，2025-01–2026-09 在冻结后跑一次。
完整评价 = 计算因子矩阵 + 训练期 RankIC/HAC/五分组 + 与因子池的最大相关与增量 IC。

```bash
python run_evolution.py --data-dir feathers --st st_mask.parquet --st-proxy st_proxy.parquet                         --budget 64 --rounds 8 --random 64 --out results/evolution
```

产出 `results/evolution/`：`EVOLUTION.md`、`candidates.csv`（全部候选、接受/拒绝原因）、`trace_*.csv`、`lineage.json`、`evolution.json`；
网页是 `evolution.html`（读 `data/evolution.json`）。

这一轮的核心结论（数字见 `EVOLUTION.md`）：三组都没能在样本外跑赢同池等权；
基线组的单点最优在样本外 IC 反而最高；Harness 与学习组提高的是"池的平均质量"而不是"最好的那一个"；
随机对照说明几十次随机搜索就足以在训练期刷出 `\|RankIC\|≈0.09`，同一批公式样本外只剩约 0.067——
所以"训练期高分"必须和搜索次数一起报。

## 局限

- 日线撮合：开盘价成交是近似，无法还原排队与盘中路径；涨跌停只用"开盘是否封板"拒单。
- ST 在 2024 年之前只能用 ±5% 涨跌幅行为推断（在 BaoStock 官方标签上验证：precision ≈ 0.84、recall ≈ 0.39），会漏判也会误判。
- 中性化只有四个代理，**没有行业和真实市值**；"残差 IC 仍为正"不等于独立 alpha。
- 期末卖不掉的残仓按收盘价计入净值，不是已变现的现金收益。
- 2021–2026 的历史在研究过程中已被反复查看，测试段不是真正的盲测。
- 分组收益是未扣费的报价统计，不能当成可执行的多空收益。

## 与第一版的关系

`legacy_v1/` 保留了最初的探索性版本（因子汇总 + 滚动 RankIC + 分组 + IPCA/树模型诊断），
本版在其基础上补齐了显著性校正、切分与方向冻结、中性化、冗余检查和带约束的账户检验。
