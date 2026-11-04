"""用 ±5% 涨跌幅行为推断 ST 状态, 并在 BaoStock 覆盖的证券上验证精度.

A 股主板 ST/*ST 的涨跌幅限制是 5%, 非 ST 是 10%(创业板/科创板 20%).
因此"过去 60 个交易日里出现多次接近 5% 的涨跌停, 且从未超过 5.5%"是
很强的 ST 信号.

产出 ../st_proxy.parquet (dates x codes, int8: 1=推断ST, 0=推断非ST)
并打印与官方逐日标签的混淆矩阵.
用法: py build_st_proxy.py
"""
import numpy as np
import pandas as pd

DATA = 'feathers'


def build_proxy(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                win_short=60, win_long=120, hit=0.048, cap=0.055, min_hits=2):
    ret = close.pct_change(fill_method=None)
    absr = ret.abs()
    # 接近 5% 的涨跌停次数 (不含 10%/20% 的票: 后者 max 会超 cap 被排除)
    hits = ((absr >= hit) & (absr <= 0.055)).rolling(win_short, min_periods=20).sum()
    maxabs = absr.rolling(win_long, min_periods=20).max()
    one_word = ((high - low).abs() < 1e-9) & (absr >= hit)     # 一字板也计一次
    hits = hits + one_word.rolling(win_short, min_periods=20).sum() * 0.0  # 已含在 hits 内
    proxy = ((maxabs <= cap) & (hits >= min_hits)).astype('int8')
    proxy[close.isna()] = 0
    return proxy


def main():
    close = pd.read_feather(f'{DATA}/close.feather').set_index('date')
    high = pd.read_feather(f'{DATA}/high.feather').set_index('date')
    low = pd.read_feather(f'{DATA}/low.feather').set_index('date')
    proxy = build_proxy(close, high, low)
    proxy.to_parquet('../st_proxy.parquet')

    off = pd.read_parquet('../st_mask.parquet')
    common = [c for c in off.columns if c in proxy.columns]
    # 只比较 2024-02 之后 (滚动窗口已满)
    d0 = off.index[0] + pd.Timedelta(days=120)
    o = off.loc[d0:, common]
    p = proxy.loc[d0:, common]
    tp = int(((o == 1) & (p == 1)).sum().sum())
    fp = int(((o == 0) & (p == 1)).sum().sum())
    fn = int(((o == 1) & (p == 0)).sum().sum())
    tn = int(((o == 0) & (p == 0)).sum().sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f'验证集: {len(common)} 只 x {len(o)} 日')
    print(f'  官方ST日 {tp+fn}, 推断ST日 {tp+fp}, 命中 {tp}')
    print(f'  precision {prec:.3f}  recall {rec:.3f}  '
          f'误标ST率 {fp/max(tn+fp,1):.4f}')
    # 按证券看: 是否至少有一天被识别
    codes_off = set(o.columns[(o == 1).any()])
    codes_prx = set(p.columns[(p == 1).any()])
    print(f'  出现过ST的证券: 官方 {len(codes_off)}, 推断 {len(codes_prx)}, '
          f'交集 {len(codes_off & codes_prx)}')
    print(f'  非ST证券被误标为ST的比例: '
          f'{len(codes_prx - codes_off)/max(len(set(o.columns)-codes_off),1):.4f}')


if __name__ == '__main__':
    main()
