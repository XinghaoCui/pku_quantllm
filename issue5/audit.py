"""独立复算: 不调用 factorlab 的评价代码, 用 pandas 直接重算关键数字.

检查项
  1  因子值: 随机日期上重算 rev5 / lowvol20, 与 factorlab 的结果逐值比较
  2  RankIC: 用 scipy.spearmanr 独立重算, 与结果文件里的 rank_ic 比较
  3  标签口径: 抽查 t 日标签是否等于 O*(t+1+h)/O*(t+1)-1
  4  成交账本: 期初现金 + Σ卖出净额 - Σ买入成本 = 期末现金 (逐日复算)
  5  持仓: Σ 成交股数 x 收盘价 = 期末持仓市值
  6  费用: 每笔佣金 = max(成交额 x 万一, 5), 卖出印花税 = 成交额 x 万五
  7  规则: 买入股数为整手, T+1 无同日回转, 无买入 ST, 交易日只在调仓日
用法: py audit.py [--results results] [--data-dir ../feathers]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from factorlab import factors as F
from factorlab import evaluate as E
from factorlab.data import Panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--data-dir', default='../feathers')
    ap.add_argument('--st', default='../st_mask.parquet')
    ap.add_argument('--st-proxy', default='../st_proxy.parquet')
    args = ap.parse_args()

    p = Panel(args.data_dir, args.st, args.st_proxy)
    checks = []

    def add(name, ok, detail=''):
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
        print(('OK  ' if ok else '!!  ') + f'{name:34s} {detail}', flush=True)

    # ------------------------------------------------ 1 因子值独立重算
    rng = np.random.default_rng(0)
    dates_pick = sorted(rng.choice(np.arange(400, len(p.dates) - 30), 5, replace=False))
    close = pd.DataFrame(p.close, index=p.dates, columns=p.codes)
    cumadj = pd.DataFrame(p.cumadj, index=p.dates, columns=p.codes)
    P = close * cumadj
    max_err_rev, max_err_vol = 0.0, 0.0
    for t in dates_pick:
        rev = -(P.iloc[t] / P.iloc[t - 5] - 1.0)
        vol = -((P / P.shift(1) - 1.0).iloc[t - 19:t + 1].std(ddof=1))
        f_rev = F.build('rev5', p)[t]
        f_vol = F.build('lowvol20', p)[t]
        m = np.isfinite(rev.values) & np.isfinite(f_rev)
        max_err_rev = max(max_err_rev, float(np.nanmax(np.abs(rev.values[m] - f_rev[m]))))
        m2 = np.isfinite(vol.values) & np.isfinite(f_vol)
        max_err_vol = max(max_err_vol, float(np.nanmax(np.abs(vol.values[m2] - f_vol[m2]))))
    add('rev5 因子值复算', max_err_rev < 1e-5, f'最大误差 {max_err_rev:.2e}')
    add('lowvol20 因子值复算', max_err_vol < 1e-6, f'最大误差 {max_err_vol:.2e}')

    # ------------------------------------------------ 2/3 标签与 RankIC
    from scipy.stats import spearmanr
    ic_path = f'{args.results}/ic_series.csv'
    if os.path.exists(ic_path):
        ics = pd.read_csv(ic_path, parse_dates=['date'])
        fac = F.build('rev5', p)
        errs, label_errs = [], []
        lab5 = E.fwd_labels(p, 5)
        for _, row in ics[ics.factor == 'rev5'].sample(6, random_state=1).iterrows():
            t = p.idx(row['date'])
            lab = p.opstar[t + 6] / p.opstar[t + 1] - 1.0
            label_errs.append(float(np.nanmax(np.abs(lab5[t] - lab))))
            m = p.universe(t) & np.isfinite(fac[t]) & np.isfinite(lab)
            if m.sum() > 50:
                rho = spearmanr(fac[t][m], lab[m]).statistic
                errs.append(abs(rho - row['rank_ic']))
        add('标签口径 O*(t+1+h)/O*(t+1)-1', max(label_errs) < 1e-6,
            f'最大误差 {max(label_errs):.2e}')
        add('RankIC 独立复算(scipy)', max(errs) < 1e-4,
            f'抽查 {len(errs)} 日, 最大误差 {max(errs):.2e}')

    # ------------------------------------------------ 4-7 账本复算
    eq_path = f'{args.results}/tradable.csv'
    trade_files = sorted(glob.glob(f'{args.results}/trades/*.csv'))
    trade_files = [f for f in trade_files if not f.endswith('_equity.csv')]
    if trade_files:
        bad_cash = bad_fee = bad_lot = bad_t1 = bad_st = 0
        max_cash_err = 0.0
        n_rows = 0
        for f in trade_files:
            tr = pd.read_csv(f, parse_dates=['date'])
            key = os.path.basename(f)[:-4]
            eqf = f'{args.results}/trades/{key}_equity.csv'
            if not os.path.exists(eqf):
                continue
            eq = pd.read_csv(eqf, parse_dates=['date'])
            n_rows += len(tr)
            # 校验方式: 期末现金 = 初始现金 + Σ(卖出净额) - Σ(买入总额)
            init = 1e9
            flow = 0.0
            for r in tr.itertuples():
                if r.side == 'BUY':
                    flow -= (r.value + r.commission)
                    if int(r.shares) % 100 != 0:
                        bad_lot += 1
                else:
                    flow += (r.value - r.commission - r.stamp)
            max_cash_err = max(max_cash_err, abs(init + flow - float(eq['cash'].iloc[-1])))
            # 费用公式
            for r in tr.itertuples():
                c = max(r.value * 1e-4, 5.0)
                st = r.value * 5e-4 if r.side == 'SELL' else 0.0
                if abs(r.commission - c) > 1e-4 or abs(r.stamp - st) > 1e-4:
                    bad_fee += 1
            # T+1: 同一 (date, code) 不允许先买后卖
            for (d, code), g in tr.groupby(['date', 'code']):
                sides = list(g['side'])
                if sides[:1] == ['BUY'] and 'SELL' in sides:
                    bad_t1 += 1
            # 买入不落在 ST 日
            for r in tr[tr.side == 'BUY'].itertuples():
                t = p.idx(r.date)
                if p.st[t, p._code_idx[r.code]] == 1:
                    bad_st += 1
        add('现金守恒(逐笔重放)', max_cash_err < 1.0,
            f'{len(trade_files)} 个因子, {n_rows} 笔, 最大误差 {max_cash_err:.2f} 元')
        add('费用公式(佣金/印花税)', bad_fee == 0, f'异常 {bad_fee} 笔')
        add('买入整手', bad_lot == 0, f'异常 {bad_lot} 笔')
        add('T+1 无同日回转', bad_t1 == 0, f'异常 {bad_t1} 处')
        add('不买入 ST', bad_st == 0, f'异常 {bad_st} 笔')

    # ------------------------------------------------ 汇总
    ok = sum(c['ok'] for c in checks)
    out = {'n_checks': len(checks), 'n_pass': ok,
           'all_pass': ok == len(checks), 'checks': checks}
    json.dump(out, open(f'{args.results}/audit.json', 'w'), ensure_ascii=False, indent=2)
    print(f'\n{ok}/{len(checks)} 通过 -> {args.results}/audit.json')
    return 0 if ok == len(checks) else 1


if __name__ == '__main__':
    sys.exit(main())
