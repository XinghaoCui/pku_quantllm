# legacy_v1 · 第五讲第一版

这是 2026-10-26 提交的第一版：因子汇总、滚动 RankIC、五分组、相关性，以及两个可解释模型诊断（IPCA 风格条件暴露、浅层树）。

保留原因：第二版重写了评价口径（改用次日开盘到开盘的标签、加了 HAC/FDR、训练/验证/测试切分与方向冻结、代理中性化和带约束的账户检验），旧结果无法与新版逐格对齐，但作为研究过程的记录仍然有用。

局限（当时已在原 README 中声明，这里再列一次）：

- 没有 Newey-West/HAC，也没有多重检验校正；
- 没有历史 ST 与上市状态过滤，股票池是"有价即入"；
- 分组价差是未扣费的描述性统计，不能当作可交易收益；
- 无训练/验证/测试切分，方向与参数在同一段数据上确定。

文件：

- `analyze.py`、`model_diagnostics.py`：第一版计算脚本
- `data/`：第一版输出（`dashboard.json`、`ic_timeseries.csv` 等）
- `tests/test_analyze.py`：第一版测试

复现方式与第一版 README 相同：把七张 Feather 放在 `issue5/feathers/`，然后
`python legacy_v1/analyze.py --data-dir ../feathers --output-dir legacy_v1/data`。
