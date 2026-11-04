"""
抓取 BaoStock 逐日 isST 标签, 生成 data/st_mask.parquet (dates x codes, int8)
范围: 2020-01-01 ~ 2026-10-16 (覆盖因子评价的全部历史)

支持增量续抓: 已存在的列不会重复请求, 失败的列下次再跑会重试.
用法: py fetch_st.py [workers]   # 数据放在 feathers/, 结果写到 st_mask.parquet
"""
import os
import sys
import time
from multiprocessing import Pool

import baostock as bs
import pandas as pd

START, END = '2020-01-01', '2026-10-16'
DATA = 'feathers'
MASK = 'st_mask.parquet'


def to_bs(code: str) -> str:
    num, ex = code.split('.')
    return ('sh.' if ex == 'SH' else 'sz.') + num


def _init():
    bs.login()


def fetch_one(code: str):
    bsc = to_bs(code)
    for attempt in range(4):
        try:
            rs = bs.query_history_k_data_plus(bsc, 'date,isST',
                                              start_date=START, end_date=END,
                                              frequency='d')
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != '0':
                raise RuntimeError(rs.error_msg)
            if not rows:                      # 空结果 = 该代码在 BaoStock 无日线
                return code, []
            return code, [(d, int(s)) for d, s in rows]
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return code, None


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    codes = [c for c in pd.read_feather(f'{DATA}/close.feather').columns if c != 'date']

    old = None
    if os.path.exists(MASK):
        old = pd.read_parquet(MASK)
        todo = [c for c in codes if c not in old.columns]
        print(f'续抓: 已有 {old.shape[1]} 列, 待抓 {len(todo)} 列', flush=True)
    else:
        todo = codes

    results = {}
    failed = []
    if todo:
        t0 = time.time()
        done = 0
        with Pool(workers, initializer=_init) as pool:
            for code, rows in pool.imap_unordered(fetch_one, todo, chunksize=4):
                done += 1
                if rows is None:
                    failed.append(code)
                else:
                    results[code] = rows
                if done % 250 == 0:
                    print(f'  {done}/{len(todo)}  ({time.time()-t0:.0f}s, 失败 {len(failed)})', flush=True)
        print(f'本轮 {time.time()-t0:.0f}s, 失败 {len(failed)}', flush=True)

    # 组装 (旧列 + 新列). 只有真正拿到数据的代码才进列集合 ——
    # 未覆盖 != 非ST, 引擎会把未覆盖的代码当作 unknown 并禁止买入.
    covered = sorted({c for c, rows in results.items() if rows} | (set(old.columns) if old is not None else set()))
    dates = sorted({d for rows in results.values() for d, _ in rows}
                   | (set(old.index.strftime('%Y-%m-%d')) if old is not None else set()))
    df = pd.DataFrame(0, index=pd.to_datetime(dates), columns=covered, dtype='int8')
    if old is not None:
        df.loc[old.index, old.columns] = old.values
    for code, rows in results.items():
        for d, s in rows:
            df.loc[pd.Timestamp(d), code] = s

    df.to_parquet(MASK)
    missing = [c for c in codes if c not in df.columns]
    print(f'st_mask shape {df.shape}, ST-day fraction {df.values.mean():.4f}')
    print(f'saved {MASK}; 仍未覆盖 {len(missing)} 列; 失败样例 {failed[:5]}')


if __name__ == '__main__':
    main()
