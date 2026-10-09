# 第三节实践课作业 · 上交所 Level-2 订单簿重建与可视化

数据：`sse50_20260923`（2026-09-23 上证50），来源见课程 Issue #6 的 Release 下载链接。
本包**不含**原始 parquet 数据（体积过大），请自行从课程 Release 下载后放入本目录即可运行重建。
可视化所需的盘口快照已导出为 `book_data.json`（202KB），**无需原始数据即可直接查看可视化界面**。

## 文件清单

| 文件 | 说明 |
|------|------|
| `orderbook.py` | 订单簿重建程序（价位消耗法），含重建/验证/导出三种模式 |
| `orderbook_vis.html` | 交互式订单簿可视化页面（单文件、离线可用） |
| `book_data.json` | 由 `snp` 快照导出的 177 帧盘口数据，供可视化使用 |
| `启动可视化.bat` | 一键启动本地服务并打开可视化页面 |
| `README.md` | 本说明 |

## 一、查看可视化界面（无需原始数据）

方式 A（推荐，Windows）：双击 `启动可视化.bat`，浏览器自动打开
`http://127.0.0.1:8765/orderbook_vis.html`，关闭黑窗口即停止服务。

方式 B（手动）：
```
py -m http.server 8765 --bind 127.0.0.1
```
然后浏览器打开 `http://127.0.0.1:8765/orderbook_vis.html`。

> 注意：请勿直接双击 html 用 `file://` 打开——浏览器 CORS 会拦截 fetch，
> 页面将回退到内置模拟数据（顶部会标注"数据源: 模拟数据"）。走本地 HTTP 才会加载真实 `book_data.json`。

可视化功能：时间轴拖动 / 自动播放（1x~10x）、买卖十档深度图、累计深度曲线、
订单簿不平衡度仪表盘与全天走势、盘口数字面板、涨跌停与价差统计。

## 二、运行订单簿重建（需原始 parquet）

将课程 Release 的三个 parquet 放入本目录后：

```
# 重建某只股票到指定时间并打印盘口
py orderbook.py --symbol 601688 --time 100000000 --levels 10

# 与交易所快照交叉验证
py orderbook.py --symbol 601688 --time 100000000 --validate

# 从快照导出可视化 JSON
py orderbook.py --symbol 601688 --export --interval 300000
```

依赖：`pandas`、`pyarrow`（`py -m pip install pandas pyarrow`）。

## 三、算法与数据说明（重要）

重建采用**价位消耗法**：将 `ord`（逐笔委托）与 `exe`（逐笔成交）按 `(Time, BizIndex)`
合并为事件流，新增委托入簿、撤单移除、成交按 BSFlag 判定主动方后从被动方价位 FIFO 消耗。

经验证，本数据集的 `ord` 文件为**不完整采样**（仅约 30% 委托）：`exe` 的 BidOrder/AskOrder
与 `ord` 的 OrderNumber 匹配率仅 0.3%，撤单引用亦无法对齐。因此逐笔重建存在量级偏差，
而 `snp` 快照（交易所每 3 秒发布的完整十档盘口）为准确 ground truth，
故可视化直接基于 `snp` 导出，保证界面数据真实可靠。
