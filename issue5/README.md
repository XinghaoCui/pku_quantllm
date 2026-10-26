# 第五讲：因子挖掘与动态评价

使用课程提供的 A 股日频宽表 Feather 数据，构造一组可解释的价格、风险与流动性代理因子，并通过交互式研究台比较其横截面预测和分组表现。

## 文件

- `analyze.py`：载入本地行情、构造因子、计算统计量并输出紧凑结果文件。
- `index.html`：中文交互仪表盘；可选择因子、预测周期、滚动窗口和日期区间，并下载当前周期指标 CSV。
- `data/dashboard.json`：可视化使用的衍生汇总数据，不含原始价格矩阵。
- `data/factor_summary.csv`、`data/ic_timeseries.csv`、`data/quantile_returns.csv`：可复查、可导出的研究结果。
- `tests/test_analyze.py`：因子与评价计算的基础测试。

## 数据

将 `open.feather`、`high.feather`、`low.feather`、`close.feather`、`amt.feather`、`num_trades.feather`、`cumadj.feather` 放在 `issue5/feathers/`。每个文件应为相同日期、相同股票列顺序的宽表，第一列为 `date`。默认读取此目录；也可以用 `--data-dir` 指定本地目录。

原始 Feather 文件可能含课程数据，不应提交到公开 fork。分析器只把派生统计写入本目录的 `data/`。

## 构造口径

因子计算使用复权收盘价 `P* = close × cumadj`；评价收益使用下一交易日复权开盘价入场：

- 动量：5、20、60、120 个交易日复权收益。
- 反转：前一交易日复权收益取负。
- 低波动：20、60 日复权日收益标准差取负。
- 低振幅：20 日对数高低价区间标准差取负。
- 流动性代理：20 日平均成交额、平均成交笔数取 `log1p`。

因子值越高均按“预期收益越高”定向，其中波动与振幅取负。信号使用 t 日收盘后已知信息；评价标签为 `close*(t+h) / open*(t+1) - 1`，不使用 t+h 之后的数据。没有行业分类和显式 ST/上市状态数据，因此本版未做行业中性化或额外股票池过滤；所有指标均是原始横截面结果，不应被误读为可交易策略回测。

## 评价维度

- Pearson IC、Spearman RankIC、均值、标准差、ICIR、正 IC 比例。
- 21/63/126/252 交易日滚动 RankIC 均值/IR；可在网页选择窗口，并按日期筛选观察衰减或反转。
- 1/5/20 日预测周期的五分组平均未来收益、Q5-Q1 价差、分组单调性、年化价差 Sharpe、价差最大回撤。
- **IPCA 风格条件暴露诊断**：用横截面特征作为条件暴露工具变量，以时间前 70% 拟合低秩 `r_it = (z_it Γ) f_t`，测试期报告预测 R² 和横截面拟合 R²，并做逐特征删除消融。
- **资产定价树风格诊断**：浅层全局切分树在训练期拟合特征交互，按时间留出评估预测 R²；逐特征删除比较其增量。

这两项为课程作业中的可解释模型诊断，不是 IPCA 论文估计程序的严格复刻，也不是完整 P-Tree/SDF 定价测试；树模型部分用于补充线性条件暴露关系下的非线性检验。预测 R² 是对严格留出收益的预测指标，另行报告的定价/横截面拟合指标不等同于策略回报。模型诊断使用固定随机种子、训练集内估计参数，具有抽样/正则化依赖，结果应视为探索性证据。
- 日度多空分组换手估算及成本敏感性。成本后价差是简化估算：`毛 Q5-Q1 - 2 × 单边成本 × 平均日换手`，未扣借券成本或冲击成本。
- 最新可用截面的因子 Spearman 相关矩阵，用于检查信息重合。

注意，多日未来收益窗口重叠，IC 和价差序列存在自相关；展示的 IR/Sharpe 仅为描述性统计，没有 Newey-West/HAC 或多重检验修正。宽表中股票历史覆盖不齐，横截面样本数随日期变化。历史有效不代表未来有效，全部发现需进一步样本外与真实成交约束验证。

## 运行

在仓库根目录执行：

```bash
uv run --with pandas --with numpy --with pyarrow --with scipy python issue5/analyze.py \
  --data-dir "../feathers" --output-dir issue5/data
uv run --with pandas --with numpy --with scipy python -m unittest discover -s issue5/tests -v
```

Windows PowerShell 示例：

```powershell
uv run --with pandas --with numpy --with pyarrow --with scipy python issue5/analyze.py `
  --data-dir "../feathers" --output-dir issue5/data
uv run --with pandas --with numpy --with scipy python -m unittest discover -s issue5/tests -v
```

数据位于本课程工作区 `issue5/feathers`，而不是公开 fork 时，可从 clone 的 issue5 目录这样指定：

```bash
uv run --with pandas --with numpy --with pyarrow --with scipy python .repo/issue5/analyze.py \
  --data-dir ../feathers --output-dir .repo/issue5/data
```

启动网页（先运行分析器生成 JSON）：

```bash
python -m http.server 8000 --directory issue5
```

浏览器访问 `http://localhost:8000`。也可以直接用浏览器打开 HTML，但某些浏览器会限制本地 `fetch`，建议使用本地 HTTP 服务。

## 复现结果

默认单边成本为 10 bp，可用 `--cost-bps` 调整。结果文件完全由本地输入生成；CSV 可供进一步检查。公共仓库仅存放代码、说明和小型派生数据，不包含课程原始行情文件。
