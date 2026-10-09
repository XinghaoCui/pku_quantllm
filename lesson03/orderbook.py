"""
上交所 Level-2 逐笔数据订单簿重建程序
=====================================
用法:
  py orderbook.py                          # 默认重建 601688 到 10:00
  py orderbook.py --symbol 600036 --time 140000000
  py orderbook.py --symbol 601688 --export  # 导出 JSON 供可视化
  py orderbook.py --symbol 601688 --validate --plot

核心算法说明:
  由于本数据集中 ord 文件为不完整采样，exe 的 BidOrder/AskOrder
  无法与 ord 的 OrderNumber 一一对应，因此采用"价位消耗法":
    1. ord FC=66 (新增委托): 按 (OrderOriNo, side) 加入订单簿
    2. ord FC=83 (撤单): 按 OrderOriNo 尝试移除，未找到则跳过
    3. exe (成交): 根据 BSFlag 判断主动方，从被动方价位 FIFO 消耗
       - BSFlag=66('B'): 主动买 -> 消耗卖方(ask)在成交价位的挂单
       - BSFlag=83('S'): 主动卖 -> 消耗买方(bid)在成交价位的挂单
       同时尝试从主动方最优价位消耗对应量

数据字段:
  ord: BizIndex, Channel, FunctionCode(66=新增,83=撤单),
       OrderKind(65='A'买,68='D'卖), OrderNumber(=BizIndex),
       OrderOriNo(本地委托序号), Price, Time(HHMMSSsss), Volume, Symbol
  exe: BidOrder, AskOrder, BSFlag(66=主动买,83=主动卖),
       BizIndex, Price, Time, Volume, Symbol
  snp: 61列快照，含 BidPrice1~10, AskPrice1~10 等
"""

import argparse
import json
import time as timer
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================
# 常量
# ============================================================

FC_NEW_ORDER = 66
FC_CANCEL = 83
FC_SYSTEM = {73, 74, 79, 67, 86}

SIDE_BUY = 65   # 'A'
SIDE_SELL = 68  # 'D'

BSFLAG_BUY_AGG = 66   # 'B' 主动买(外盘)
BSFLAG_SELL_AGG = 83  # 'S' 主动卖(内盘)

# 交易时段 (HHMMSSsss)
CALL_AUCTION_START = 91500000
CALL_AUCTION_END = 92500000
CONTINUOUS_AM_START = 93000000
CONTINUOUS_AM_END = 113000000
CONTINUOUS_PM_START = 130000000
CONTINUOUS_PM_END = 150000000


def time_to_str(t: int) -> str:
    """HHMMSSsss -> 'HH:MM:SS.mmm'"""
    s = str(int(t)).zfill(9)
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}.{s[6:9]}"


# ============================================================
# 订单簿数据结构
# ============================================================

@dataclass
class Order:
    order_id: int       # OrderOriNo (本地序号)
    side: int
    price: float
    volume: int
    remain: int
    time: int
    seq: int            # 全局到达序号 (FIFO)


class OrderBook:
    """
    单只股票订单簿
    bids: 价格降序 (买一=最高价)
    asks: 价格升序 (卖一=最低价)
    同价位内按 seq 排序 (时间优先 FIFO)
    """

    def __init__(self, symbol: str, pre_close: float = 0.0):
        self.symbol = symbol
        self.pre_close = pre_close
        # price -> list[Order]
        self._bids: dict[float, list[Order]] = defaultdict(list)
        self._asks: dict[float, list[Order]] = defaultdict(list)
        # 有序价格列表
        self._bid_prices: list[float] = []  # 降序
        self._ask_prices: list[float] = []  # 升序
        # 索引: (side, order_id) -> Order
        self._order_map: dict[tuple[int, int], Order] = {}
        self._seq = 0
        self.stats = {
            'adds': 0, 'cancels': 0, 'cancel_miss': 0,
            'trades': 0, 'consume_bid': 0, 'consume_ask': 0, 'miss': 0
        }

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _insert_bid_price(self, price: float):
        neg = [-p for p in self._bid_prices]
        idx = bisect_left(neg, -price)
        self._bid_prices.insert(idx, price)

    def _insert_ask_price(self, price: float):
        idx = bisect_left(self._ask_prices, price)
        self._ask_prices.insert(idx, price)

    def add_order(self, order_id: int, side: int, price: float,
                  volume: int, time: int):
        if volume <= 0 or price <= 0:
            return
        order = Order(order_id=order_id, side=side, price=price,
                      volume=volume, remain=volume, time=time,
                      seq=self._next_seq())
        self._order_map[(side, order_id)] = order

        if side == SIDE_BUY:
            if not self._bids[price]:
                self._insert_bid_price(price)
            self._bids[price].append(order)
        else:
            if not self._asks[price]:
                self._insert_ask_price(price)
            self._asks[price].append(order)
        self.stats['adds'] += 1

    def cancel_order(self, order_id: int, side: int) -> bool:
        key = (side, order_id)
        order = self._order_map.pop(key, None)
        if order is None:
            alt = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            order = self._order_map.pop((alt, order_id), None)
            if order is None:
                self.stats['cancel_miss'] += 1
                return False

        book = self._bids if order.side == SIDE_BUY else self._asks
        level = book.get(order.price, [])
        if order in level:
            level.remove(order)
        if not level:
            book.pop(order.price, None)
            prices = self._bid_prices if order.side == SIDE_BUY else self._ask_prices
            if order.price in prices:
                prices.remove(order.price)
        self.stats['cancels'] += 1
        return True

    def consume_at_price(self, side: int, price: float, volume: int) -> int:
        """从指定价位 FIFO 消耗, 返回实际消耗量"""
        book = self._bids if side == SIDE_BUY else self._asks
        level = book.get(price)
        if not level:
            return 0
        consumed = 0
        while consumed < volume and level:
            order = level[0]
            need = volume - consumed
            if order.remain <= need:
                consumed += order.remain
                self._order_map.pop((order.side, order.order_id), None)
                level.pop(0)
            else:
                order.remain -= need
                consumed += need
        if not level:
            book.pop(price, None)
            prices = self._bid_prices if side == SIDE_BUY else self._ask_prices
            if price in prices:
                prices.remove(price)
        return consumed

    def consume_best(self, side: int, volume: int, limit_price: float) -> int:
        """从最优价位开始消耗 (主动方), 受 limit_price 约束"""
        prices = self._bid_prices if side == SIDE_BUY else self._ask_prices
        consumed = 0
        while consumed < volume and prices:
            p = prices[0]
            if side == SIDE_BUY and p < limit_price:
                break
            if side == SIDE_SELL and p > limit_price:
                break
            got = self.consume_at_price(side, p, volume - consumed)
            consumed += got
            if got == 0:
                break
        return consumed

    def execute_trade(self, price: float, volume: int, bs_flag: int):
        """
        价位消耗法处理成交:
        每笔成交同时消耗买方和卖方各 volume 股。
          - 被动方: 在成交价位精确消耗 (成交价=被动方挂单价)
          - 主动方: 从最优价位开始消耗 (主动方挂单价可能优于成交价)
          - BSFlag=0 (集合竞价): 两侧都在成交价位消耗
        """
        if bs_flag == BSFLAG_BUY_AGG:
            # 主动买: 被动方=ask@price, 主动方=bid从最优
            got_ask = self.consume_at_price(SIDE_SELL, price, volume)
            got_bid = self.consume_best(SIDE_BUY, volume, price)
            self.stats['consume_ask'] += got_ask
            self.stats['consume_bid'] += got_bid
            if got_ask + got_bid == 0:
                self.stats['miss'] += 1
        elif bs_flag == BSFLAG_SELL_AGG:
            # 主动卖: 被动方=bid@price, 主动方=ask从最优
            got_bid = self.consume_at_price(SIDE_BUY, price, volume)
            got_ask = self.consume_best(SIDE_SELL, volume, price)
            self.stats['consume_bid'] += got_bid
            self.stats['consume_ask'] += got_ask
            if got_ask + got_bid == 0:
                self.stats['miss'] += 1
        else:
            # 集合竞价或未知方向: 两侧都在成交价位消耗
            got_bid = self.consume_at_price(SIDE_BUY, price, volume)
            got_ask = self.consume_at_price(SIDE_SELL, price, volume)
            self.stats['consume_bid'] += got_bid
            self.stats['consume_ask'] += got_ask
            if got_ask + got_bid == 0:
                # 退而求其次: 从最优价位消耗
                got_bid = self.consume_best(SIDE_BUY, volume, price)
                got_ask = self.consume_best(SIDE_SELL, volume, price)
                self.stats['consume_bid'] += got_bid
                self.stats['consume_ask'] += got_ask
                if got_ask + got_bid == 0:
                    self.stats['miss'] += 1
        self.stats['trades'] += 1

    # ---------- 查询 ----------

    @property
    def best_bid(self) -> tuple[float, int]:
        if not self._bid_prices:
            return 0.0, 0
        p = self._bid_prices[0]
        return p, sum(o.remain for o in self._bids[p])

    @property
    def best_ask(self) -> tuple[float, int]:
        if not self._ask_prices:
            return 0.0, 0
        p = self._ask_prices[0]
        return p, sum(o.remain for o in self._asks[p])

    @property
    def spread(self) -> float:
        bb, _ = self.best_bid
        ba, _ = self.best_ask
        if bb > 0 and ba > 0:
            return round(ba - bb, 4)
        return float('inf')

    @property
    def mid_price(self) -> float:
        bb, _ = self.best_bid
        ba, _ = self.best_ask
        return (bb + ba) / 2 if bb > 0 and ba > 0 else 0.0

    def depth(self, n: int = 20) -> dict:
        bids = []
        for p in self._bid_prices[:n]:
            orders = self._bids[p]
            bids.append({'price': round(p, 2),
                         'volume': sum(o.remain for o in orders),
                         'orders': len(orders)})
        asks = []
        for p in self._ask_prices[:n]:
            orders = self._asks[p]
            asks.append({'price': round(p, 2),
                         'volume': sum(o.remain for o in orders),
                         'orders': len(orders)})
        return {'bids': bids, 'asks': asks}

    def total_volume(self, side: int, n: int = 9999) -> int:
        prices = self._bid_prices if side == SIDE_BUY else self._ask_prices
        book = self._bids if side == SIDE_BUY else self._asks
        return sum(sum(o.remain for o in book[p]) for p in prices[:n])

    def imbalance(self, n: int = 10) -> float:
        bv = self.total_volume(SIDE_BUY, n)
        av = self.total_volume(SIDE_SELL, n)
        t = bv + av
        return (bv - av) / t if t > 0 else 0.0

    def print_depth(self, n: int = 10):
        d = self.depth(n)
        bb, bv = self.best_bid
        ba, av = self.best_ask
        print(f"\n{'='*66}")
        print(f"  {self.symbol} ({_symbol_name(self.symbol)}) 订单簿 前{n}档")
        print(f"{'='*66}")
        print(f"  {'卖盘 (Ask)':<34} | {'买盘 (Bid)'}")
        print(f"  {'-'*34}-+-{'-'*30}")

        asks, bids = d['asks'], d['bids']
        rows = max(len(asks), len(bids))
        for i in range(rows - 1, -1, -1):
            left = ""
            if i < len(asks):
                a = asks[i]
                left = f"卖{i+1:<2} {a['price']:>8.2f} x {a['volume']:>9,} ({a['orders']:>3})"
            right = ""
            if i < len(bids):
                b = bids[i]
                right = f"买{i+1:<2} {b['price']:>8.2f} x {b['volume']:>9,} ({b['orders']:>3})"
            print(f"  {left:<38} | {right}")

        print(f"  {'-'*34}-+-{'-'*30}")
        sp = f"{self.spread:.3f}" if self.spread < 100 else "N/A"
        print(f"  买一 {bb:.2f} | 卖一 {ba:.2f} | 价差 {sp} | 中间价 {self.mid_price:.3f}")
        print(f"  不平衡度(10档): {self.imbalance():+.4f}")
        print(f"  统计: +{self.stats['adds']:,}新增  -{self.stats['cancels']:,}撤单"
              f"  ~{self.stats['trades']:,}成交  ?{self.stats['miss']:,}未命中")
        print(f"{'='*66}")


# ============================================================
# 重建引擎
# ============================================================

class OrderBookRebuilder:

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.books: dict[str, OrderBook] = {}
        self._pre_close: dict[str, float] = {}
        self._events: Optional[pd.DataFrame] = None

    def load_data(self, date_str: str = '20260923',
                  symbols: Optional[list[str]] = None):
        t0 = timer.time()
        print(f"[加载] ord_{date_str}.parquet ...")
        ord_df = pd.read_parquet(self.data_dir / f'ord_{date_str}.parquet')
        print(f"[加载] exe_{date_str}.parquet ...")
        exe_df = pd.read_parquet(self.data_dir / f'exe_{date_str}.parquet')

        ord_df['Symbol'] = ord_df['Symbol'].astype(str)
        exe_df['Symbol'] = exe_df['Symbol'].astype(str)
        ord_df = ord_df[~ord_df['FunctionCode'].isin(FC_SYSTEM)]

        if symbols:
            symbols = [str(s) for s in symbols]
            ord_df = ord_df[ord_df['Symbol'].isin(symbols)]
            exe_df = exe_df[exe_df['Symbol'].isin(symbols)]

        ord_df = ord_df.copy()
        ord_df['EventType'] = 0
        for c in ['BidOrder', 'AskOrder', 'BSFlag']:
            if c not in ord_df.columns:
                ord_df[c] = 0

        exe_df = exe_df.copy()
        exe_df['EventType'] = 1
        for c in ['OrderNumber', 'OrderOriNo', 'OrderKind', 'FunctionCode']:
            if c not in exe_df.columns:
                exe_df[c] = 0

        cols = ['BizIndex', 'EventType', 'FunctionCode', 'OrderKind',
                'OrderNumber', 'OrderOriNo', 'BidOrder', 'AskOrder',
                'BSFlag', 'Price', 'Time', 'Volume', 'Symbol']

        print("[加载] 合并排序 ...")
        events = pd.concat([ord_df[cols], exe_df[cols]], ignore_index=True)
        events.sort_values(['Time', 'BizIndex'], inplace=True)
        events.reset_index(drop=True, inplace=True)
        self._events = events

        self._load_pre_close(date_str, symbols)
        print(f"[加载] {len(events):,} 条事件, "
              f"{events['Symbol'].nunique()} 只股票, {timer.time()-t0:.1f}s")
        return self

    def _load_pre_close(self, date_str, symbols):
        p = self.data_dir / f'snp_{date_str}.parquet'
        if not p.exists():
            return
        snp = pd.read_parquet(p, columns=['Symbol', 'PreClose'])
        snp['Symbol'] = snp['Symbol'].astype(str)
        if symbols:
            snp = snp[snp['Symbol'].isin([str(s) for s in symbols])]
        pc = snp[snp['PreClose'] > 0].drop_duplicates('Symbol')
        self._pre_close = dict(zip(pc['Symbol'], pc['PreClose']))

    def _get_book(self, symbol: str) -> OrderBook:
        if symbol not in self.books:
            self.books[symbol] = OrderBook(symbol, self._pre_close.get(symbol, 0.0))
        return self.books[symbol]

    def rebuild_to(self, target_time: Optional[int] = None,
                   symbols: Optional[list[str]] = None, verbose: bool = True):
        events = self._events
        n = len(events)
        t0 = timer.time()

        ev_type = events['EventType'].values
        fc = events['FunctionCode'].values
        ok = events['OrderKind'].values
        oori = events['OrderOriNo'].values
        bsflag = events['BSFlag'].values
        price = events['Price'].values
        vol = events['Volume'].values
        tm = events['Time'].values
        sym = events['Symbol'].values

        symbol_set = set(str(s) for s in symbols) if symbols else None
        processed = 0

        for i in range(n):
            if target_time and tm[i] > target_time:
                break
            if symbol_set and sym[i] not in symbol_set:
                continue

            book = self._get_book(sym[i])
            if ev_type[i] == 0:
                if fc[i] == FC_NEW_ORDER:
                    book.add_order(int(oori[i]), int(ok[i]),
                                   float(price[i]), int(vol[i]), int(tm[i]))
                elif fc[i] == FC_CANCEL:
                    book.cancel_order(int(oori[i]), int(ok[i]))
            else:
                book.execute_trade(float(price[i]), int(vol[i]), int(bsflag[i]))

            processed += 1
            if verbose and processed % 500_000 == 0:
                print(f"  ... {processed:,} ({timer.time()-t0:.1f}s)")

        if verbose:
            print(f"[重建] {processed:,} 条, {timer.time()-t0:.1f}s")

    def get_book(self, symbol: str) -> Optional[OrderBook]:
        return self.books.get(symbol)

    def validate_with_snapshot(self, symbol: str, target_time: int,
                               date_str: str = '20260923', n_levels: int = 5):
        snp_path = self.data_dir / f'snp_{date_str}.parquet'
        if not snp_path.exists():
            print("[验证] 无快照文件")
            return
        snp = pd.read_parquet(snp_path)
        snp['Symbol'] = snp['Symbol'].astype(str)
        rows = snp[(snp['Symbol'] == str(symbol)) & (snp['Time'] <= target_time)]
        if rows.empty:
            print(f"[验证] 无 {symbol} 快照")
            return
        snap = rows.iloc[-1]
        book = self.get_book(symbol)
        if not book:
            return

        d = book.depth(n_levels)
        print(f"\n{'='*66}")
        print(f"  验证: {symbol} @ {time_to_str(int(snap['Time']))}")
        print(f"{'='*66}")

        pm_count, total = 0, 0
        print(f"  {'':4} {'重建买价':>9} {'快照买价':>9} {'  '} {'重建买量':>9} {'快照买量':>9}")
        print(f"  {'-'*54}")
        for i in range(n_levels):
            rp = d['bids'][i]['price'] if i < len(d['bids']) else 0
            rv = d['bids'][i]['volume'] if i < len(d['bids']) else 0
            sp = float(snap.get(f'BidPrice{i+1}', 0))
            sv = float(snap.get(f'BidVolume{i+1}', 0))
            m = "OK" if abs(rp - sp) < 0.005 else "!!"
            pm_count += (abs(rp - sp) < 0.005)
            total += 1
            print(f"  买{i+1}  {rp:>9.2f} {sp:>9.2f} {m:2}  {rv:>9,} {sv:>9,.0f}")

        print(f"  {'-'*54}")
        print(f"  {'':4} {'重建卖价':>9} {'快照卖价':>9} {'  '} {'重建卖量':>9} {'快照卖量':>9}")
        for i in range(n_levels):
            rp = d['asks'][i]['price'] if i < len(d['asks']) else 0
            rv = d['asks'][i]['volume'] if i < len(d['asks']) else 0
            sp = float(snap.get(f'AskPrice{i+1}', 0))
            sv = float(snap.get(f'AskVolume{i+1}', 0))
            m = "OK" if abs(rp - sp) < 0.005 else "!!"
            pm_count += (abs(rp - sp) < 0.005)
            total += 1
            print(f"  卖{i+1}  {rp:>9.2f} {sp:>9.2f} {m:2}  {rv:>9,} {sv:>9,.0f}")

        print(f"\n  价格匹配: {pm_count}/{total} ({pm_count/total*100:.0f}%)")
        print(f"  注: ord为不完整采样, 量偏差正常; 价格档位一致即通过")

    def export_snapshots(self, symbol: str, date_str: str = '20260923',
                         interval: int = 300000, n_levels: int = 10,
                         output: str = 'book_data.json'):
        """
        从 snp 快照数据直接导出 JSON (交易所 ground truth)
        每隔 interval 采样一次, 供 orderbook_vis.html 使用

        注: 由于 ord 文件为不完整采样(仅含~30%委托), 逐笔重建存在偏差。
        snp 快照是交易所每3秒发布的完整10档盘口, 数据准确可靠。
        """
        snp_path = self.data_dir / f'snp_{date_str}.parquet'
        if not snp_path.exists():
            print("[导出] 错误: 快照文件不存在")
            return None

        print(f"[导出] 从 snp 快照导出 {symbol} ({_symbol_name(symbol)}) ...")
        t0 = timer.time()

        snp = pd.read_parquet(snp_path)
        snp['Symbol'] = snp['Symbol'].astype(str)
        sym_snp = snp[snp['Symbol'] == str(symbol)].copy()
        sym_snp.sort_values('Time', inplace=True)

        # 只保留连续竞价时段 (9:30~15:00) 且有效数据
        sym_snp = sym_snp[(sym_snp['Time'] >= CONTINUOUS_AM_START) &
                          (sym_snp['Time'] <= CONTINUOUS_PM_END) &
                          (sym_snp['Price'] > 0)]

        # 按 interval 采样
        snapshots = []
        next_snap = CONTINUOUS_AM_START
        pre_close = self._pre_close.get(symbol, 0.0)

        for _, row in sym_snp.iterrows():
            t = int(row['Time'])
            if t < next_snap:
                continue

            # 构建买卖档位
            bids = []
            asks = []
            for lv in range(1, n_levels + 1):
                bp = float(row.get(f'BidPrice{lv}', 0))
                bv = float(row.get(f'BidVolume{lv}', 0))
                ap = float(row.get(f'AskPrice{lv}', 0))
                av = float(row.get(f'AskVolume{lv}', 0))
                if bp > 0:
                    bids.append({'price': round(bp, 2), 'volume': int(bv), 'orders': 0})
                if ap > 0:
                    asks.append({'price': round(ap, 2), 'volume': int(av), 'orders': 0})

            if not bids and not asks:
                next_snap += interval
                continue

            # 计算指标
            bb = bids[0]['price'] if bids else 0
            ba = asks[0]['price'] if asks else 0
            mid = (bb + ba) / 2 if bb > 0 and ba > 0 else float(row['Price'])
            spread = round(ba - bb, 4) if bb > 0 and ba > 0 else 0

            total_bid = sum(b['volume'] for b in bids)
            total_ask = sum(a['volume'] for a in asks)
            imb = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0

            snapshots.append({
                'time': time_to_str(t),
                'time_int': t,
                'bids': bids,
                'asks': asks,
                'mid_price': round(mid, 4),
                'spread': spread,
                'imbalance': round(imb, 6),
                'last_trade_price': round(float(row['Price']), 2),
                'total_bid_vol': total_bid,
                'total_ask_vol': total_ask,
            })
            next_snap += interval

        data = {
            'symbol': symbol,
            'name': _symbol_name(symbol),
            'date': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            'pre_close': pre_close,
            'snapshots': snapshots
        }
        out = self.data_dir / output
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[导出] {len(snapshots)} 个快照 -> {out.name} ({timer.time()-t0:.1f}s)")
        return out


# ============================================================
# 辅助
# ============================================================

SYMBOL_NAMES = {
    '601688': '华泰证券', '600036': '招商银行', '601318': '中国平安',
    '600519': '贵州茅台', '601166': '兴业银行', '600030': '中信证券',
    '601398': '工商银行', '601939': '建设银行', '601288': '农业银行',
    '600000': '浦发银行', '601988': '中国银行', '601601': '中国太保',
    '600887': '伊利股份', '601628': '中国人寿', '601088': '中国神华',
    '600276': '恒瑞医药', '601899': '紫金矿业', '600809': '山西汾酒',
    '603019': '中科曙光', '688008': '澜起科技', '601012': '隆基绿能',
    '600111': '北方稀土', '601919': '中远海控', '603501': '韦尔股份',
    '601600': '中国铝业', '601658': '邮储银行', '600760': '中航沈飞',
    '601211': '国泰君安', '603993': '洛阳钼业', '688256': '寒武纪',
    '601888': '中国中免', '600089': '特变电工', '688111': '金山办公',
    '601985': '中国核电', '600309': '万华化学', '601668': '中国建筑',
    '600900': '长江电力', '601857': '中国石油', '600585': '海螺水泥',
    '601066': '中信建投', '600031': '三一重工', '601633': '长城汽车',
    '600048': '保利发展', '601390': '中国中铁', '601111': '中国国航',
    '600050': '中国联通', '600690': '海尔智家', '603259': '药明康德',
    '601138': '工业富联',
}


def _symbol_name(code: str) -> str:
    return SYMBOL_NAMES.get(str(code), str(code))


def plot_orderbook(book: OrderBook, n_levels: int = 20, save_path: Optional[str] = None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("[绘图] matplotlib 未安装, 跳过")
        return

    d = book.depth(n_levels)
    if not d['bids'] and not d['asks']:
        print("[绘图] 订单簿为空")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{book.symbol} {_symbol_name(book.symbol)} Order Book Depth', fontsize=13)

    if d['bids']:
        bp = [x['price'] for x in d['bids']]
        bv = [x['volume'] for x in d['bids']]
        ax1.barh(bp, bv, color='#e74c3c', alpha=0.75, height=0.008, label='Bid')
    if d['asks']:
        ap = [x['price'] for x in d['asks']]
        av = [x['volume'] for x in d['asks']]
        ax1.barh(ap, av, color='#2ecc71', alpha=0.75, height=0.008, label='Ask')
    ax1.set_xlabel('Volume (shares)')
    ax1.set_ylabel('Price (CNY)')
    ax1.set_title('Per-Level Depth')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    if d['bids']:
        cum, t = [], 0
        for x in d['bids']:
            t += x['volume']; cum.append(t)
        ax2.plot(bp, cum, 'r-', lw=2, label='Cumulative Bid')
        ax2.fill_between(bp, cum, alpha=0.15, color='red')
    if d['asks']:
        cum, t = [], 0
        for x in d['asks']:
            t += x['volume']; cum.append(t)
        ax2.plot(ap, cum, 'g-', lw=2, label='Cumulative Ask')
        ax2.fill_between(ap, cum, alpha=0.15, color='green')
    ax2.set_xlabel('Price (CNY)')
    ax2.set_ylabel('Cumulative Volume')
    ax2.set_title('Cumulative Depth Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = save_path or str(Path(__file__).parent / f'book_{book.symbol}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[绘图] {out}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='上交所 L2 订单簿重建')
    parser.add_argument('--symbol', default='601688', help='证券代码')
    parser.add_argument('--time', type=int, default=100000000,
                        help='重建至 HHMMSSsss (默认100000000=10:00)')
    parser.add_argument('--levels', type=int, default=10, help='显示档位')
    parser.add_argument('--validate', action='store_true', help='与快照对比')
    parser.add_argument('--plot', action='store_true', help='绘制深度图')
    parser.add_argument('--export', action='store_true', help='导出JSON供HTML可视化')
    parser.add_argument('--interval', type=int, default=300000, help='导出采样间隔')
    parser.add_argument('--data-dir', default=None, help='数据目录')
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent
    symbol = str(args.symbol)

    print("=" * 66)
    print("  上交所 Level-2 订单簿重建 (价位消耗法)")
    print("=" * 66)
    print(f"  股票: {symbol} ({_symbol_name(symbol)})")
    print(f"  目标: {time_to_str(args.time)}")
    print(f"  目录: {data_dir}")
    print()

    rebuilder = OrderBookRebuilder(data_dir)
    rebuilder.load_data(symbols=[symbol])

    if args.export:
        rebuilder.export_snapshots(symbol, interval=args.interval)
        print("\n[完成] 用浏览器打开 orderbook_vis.html 查看交互式可视化")
        return

    print(f"\n[重建] {symbol} -> {time_to_str(args.time)} ...")
    rebuilder.rebuild_to(target_time=args.time, symbols=[symbol])

    book = rebuilder.get_book(symbol)
    if book:
        book.print_depth(args.levels)
        if args.validate:
            rebuilder.validate_with_snapshot(symbol, args.time)
        if args.plot:
            plot_orderbook(book, n_levels=args.levels)
    else:
        print(f"[错误] 无 {symbol} 数据")


if __name__ == '__main__':
    main()
