"""Build factor metrics and compact dashboard data from local wide Feather panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

INPUTS = ("open", "close", "high", "low", "amt", "num_trades", "cumadj")
HORIZONS = (1, 5, 20)
ROLLING_WINDOWS = (21, 63, 126, 252)
Q = 5
DEFAULT_COST_BPS = 10.0
FACTOR_INFO = {
    "mom_5": ("5日动量", "5日复权收益", "趋势延续"),
    "mom_20": ("20日动量", "20日复权收益", "趋势延续"),
    "mom_60": ("60日动量", "60日复权收益", "趋势延续"),
    "mom_120": ("120日动量", "120日复权收益", "趋势延续"),
    "reversal_1": ("1日反转", "前一交易日复权收益取负", "短期反转"),
    "vol_20": ("20日低波动", "20日复权日收益波动率取负", "风险/低波动"),
    "vol_60": ("60日低波动", "60日复权日收益波动率取负", "风险/低波动"),
    "range_20": ("20日低振幅", "20日对数高低价区间波动取负", "风险/振幅"),
    "liq_amount_20": ("20日成交额", "20日平均成交额对数", "流动性代理"),
    "liq_trades_20": ("20日成交笔数", "20日平均成交笔数对数", "流动性代理"),
}


def load_panels(data_dir: Path) -> dict[str, pd.DataFrame]:
    panels = {}
    for name in INPUTS:
        frame = pd.read_feather(data_dir / f"{name}.feather").set_index("date")
        frame.index = pd.to_datetime(frame.index)
        panels[name] = frame.astype(np.float64)
    base = panels["close"]
    for name, frame in panels.items():
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"{name}.feather does not align with close.feather")
    return panels


def build_factors(panels: dict[str, pd.DataFrame]):
    close = panels["close"] * panels["cumadj"]
    daily = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    ranges = np.log(panels["high"].where(panels["high"] > 0) / panels["low"].where(panels["low"] > 0))
    ranges = ranges.replace([np.inf, -np.inf], np.nan)
    factors = {
        "mom_5": close.pct_change(5, fill_method=None),
        "mom_20": close.pct_change(20, fill_method=None),
        "mom_60": close.pct_change(60, fill_method=None),
        "mom_120": close.pct_change(120, fill_method=None),
        "reversal_1": -daily,
        "vol_20": -daily.rolling(20, min_periods=18).std(),
        "vol_60": -daily.rolling(60, min_periods=54).std(),
        "range_20": -ranges.rolling(20, min_periods=18).std(),
        "liq_amount_20": np.log1p(panels["amt"].where(panels["amt"] >= 0).rolling(20, min_periods=15).mean()),
        "liq_trades_20": np.log1p(panels["num_trades"].where(panels["num_trades"] >= 0).rolling(20, min_periods=15).mean()),
    }
    entry = panels["open"] * panels["cumadj"]
    forward = {}
    for h in HORIZONS:
        future = close.shift(-h) / entry.shift(-1) - 1.0
        tradable = (panels["amt"].shift(-1) > 0) & (panels["amt"].shift(-h) > 0)
        forward[f"{h}d"] = future.where(tradable)
    return factors, forward


def _corr(x: np.ndarray, y: np.ndarray, ranked=False) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 30:
        return np.nan
    x, y = x[valid], y[valid]
    if ranked:
        x, y = rankdata(x, method="average"), rankdata(y, method="average")
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den > 0 else np.nan


def _max_drawdown(returns: np.ndarray) -> float:
    returns = returns[np.isfinite(returns)]
    if not returns.size:
        return np.nan
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peak - 1))


def evaluate(factors: dict[str, pd.DataFrame], forward: dict[str, pd.DataFrame], cost_bps=DEFAULT_COST_BPS):
    dates = next(iter(factors.values())).index
    ic_rows, quant_rows, summary = [], [], []
    for name, signal in factors.items():
        values = signal.to_numpy()
        valid_rows = np.isfinite(values).sum(axis=1) >= 100
        top = np.zeros(values.shape, dtype=bool)
        bottom = np.zeros(values.shape, dtype=bool)
        for i in np.flatnonzero(valid_rows):
            valid = np.isfinite(values[i])
            lo, hi = np.nanquantile(values[i, valid], [0.2, 0.8])
            top[i] = valid & (values[i] >= hi)
            bottom[i] = valid & (values[i] <= lo)
        turnover = np.full(len(dates), np.nan)
        eligible = valid_rows[1:] & valid_rows[:-1]
        top_overlap = (top[1:] & top[:-1]).sum(axis=1) / np.maximum(top[1:].sum(axis=1), 1)
        bottom_overlap = (bottom[1:] & bottom[:-1]).sum(axis=1) / np.maximum(bottom[1:].sum(axis=1), 1)
        turnover[1:] = np.where(eligible, 1 - 0.5 * (top_overlap + bottom_overlap), np.nan)
        for horizon, ret_df in forward.items():
            h = int(horizon[:-1])
            ret = ret_df.to_numpy()
            ic, ric = np.full(len(dates), np.nan), np.full(len(dates), np.nan)
            qdaily = np.full((len(dates), Q), np.nan)
            for i in range(len(dates)):
                x, y = values[i], ret[i]
                ic[i], ric[i] = _corr(x, y), _corr(x, y, ranked=True)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 100:
                    continue
                ranks = rankdata(x[mask], method="average")
                labels = np.minimum((ranks * Q / (mask.sum() + 1e-12)).astype(int), Q - 1)
                qdaily[i] = [np.mean(y[mask][labels == q]) if np.any(labels == q) else np.nan for q in range(Q)]
            ic_s, ric_s = pd.Series(ic), pd.Series(ric)
            qmeans = np.nanmean(qdaily, axis=0)
            spread = qdaily[:, -1] - qdaily[:, 0]
            clean = spread[np.isfinite(spread)]
            std_ic, std_ric = ic_s.std(), ric_s.std()
            avg_turn = float(np.nanmean(turnover))
            net = pd.Series(spread) - 2 * (cost_bps / 10000) * pd.Series(turnover)
            h = int(horizon[:-1])
            summary.append({
                "factor": name, "name": FACTOR_INFO[name][0], "category": FACTOR_INFO[name][2], "horizon": horizon,
                "ic_mean": float(ic_s.mean()), "ic_std": float(std_ic), "icir": float(ic_s.mean() / std_ic) if std_ic else None,
                "ic_positive_ratio": float((ic_s.dropna() > 0).mean()),
                "rank_ic_mean": float(ric_s.mean()), "rank_ic_std": float(std_ric),
                "rank_icir": float(ric_s.mean() / std_ric) if std_ric else None,
                "rank_ic_positive_ratio": float((ric_s.dropna() > 0).mean()),
                **{f"q{q+1}_return": float(qmeans[q]) for q in range(Q)},
                "q5_q1_spread": float(np.nanmean(spread)),
                "monotonicity": float(np.corrcoef(np.arange(Q), qmeans)[0, 1]) if np.isfinite(qmeans).all() else None,
                "spread_sharpe_annualized": float(np.nanmean(clean) / np.nanstd(clean, ddof=1) * np.sqrt(252 / h))
                    if len(clean) > 1 and np.nanstd(clean, ddof=1) > 0 else None,
                "spread_max_drawdown": _max_drawdown(clean), "avg_daily_turnover": avg_turn,
                "cost_bps_one_way": cost_bps, "net_spread_mean": float(net.mean()), "observations": int(ic_s.notna().sum()),
            })
            for i, date in enumerate(dates):
                ic_rows.append({"date": date.strftime("%Y-%m-%d"), "factor": name, "horizon": horizon,
                                "ic": float(ic[i]) if np.isfinite(ic[i]) else None,
                                "rank_ic": float(ric[i]) if np.isfinite(ric[i]) else None,
                                "turnover": float(turnover[i]) if np.isfinite(turnover[i]) else None})
            for q in range(Q):
                quant_rows.append({"factor": name, "horizon": horizon, "quantile": q + 1, "mean_return": float(qmeans[q])})

    # The dashboard computes date-filtered rolling lines client-side to keep the JSON compact.
    rolling_rows = []
    latest_corr = pd.DataFrame({name: frame.iloc[-1] for name, frame in factors.items()}).corr(method="spearman", min_periods=30)
    definitions = [{"id": key, "name": v[0], "formula": v[1], "category": v[2], "orientation": "higher is stronger"} for key, v in FACTOR_INFO.items()]
    return summary, ic_rows, quant_rows, rolling_rows, latest_corr, definitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "feathers")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = load_panels(args.data_dir)
    factors, forward = build_factors(panels)
    summary, ic, quantiles, rolling, corr, definitions = evaluate(factors, forward, args.cost_bps)
    from model_diagnostics import run_model_diagnostics
    model_diagnostics = run_model_diagnostics(factors, forward)
    payload = {
        "meta": {"start": str(next(iter(factors.values())).index.min().date()),
                 "end": str(next(iter(factors.values())).index.max().date()),
                 "n_dates": len(next(iter(factors.values())).index), "n_assets": next(iter(factors.values())).shape[1],
                 "horizons": list(forward), "rolling_windows": list(ROLLING_WINDOWS), "cost_bps": args.cost_bps,
                 "notes": ["横截面未做行业中性化；输入未提供行业分类。",
                           "t日收盘后形成信号，预测从t到t+h的复权收益；信号不读取未来数据。",
                           "多日未来收益有重叠；ICIR和Sharpe未做序列相关显著性修正。",
                           "成本为多空两腿的单边换手近似，不含冲击、涨跌停、停牌及融券约束。"]},
        "factors": definitions, "summary": summary, "ic_series": ic, "quantiles": quantiles,
        "rolling": rolling, "model_diagnostics": model_diagnostics,
        "factor_correlation": {"factors": list(corr.columns), "matrix": corr.fillna(0).round(4).values.tolist()},
    }
    out = args.output_dir / "dashboard.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    pd.DataFrame(summary).to_csv(args.output_dir / "factor_summary.csv", index=False)
    pd.DataFrame(ic).to_csv(args.output_dir / "ic_timeseries.csv", index=False)
    pd.DataFrame(quantiles).to_csv(args.output_dir / "quantile_returns.csv", index=False)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes); {len(summary)} factor/horizon rows")


if __name__ == "__main__":
    main()
