"""Fast, explicit IPCA-inspired and pricing-tree-inspired diagnostics.

The diagnostics use a strict chronological split and fixed training estimates. They
are educational model-based evaluation dimensions, not exact reproductions of the
published IPCA or P-Tree estimators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ASSETS = 500
DATE_STEP = 5
TRAIN_FRACTION = 0.70
RIDGE = 1e-3
IPCA_RANK = 3


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _sample_panel(factors: dict[str, pd.DataFrame], returns: pd.DataFrame):
    names = list(factors)
    dates = returns.index
    date_ids = np.arange(120, max(121, len(dates) - 1), DATE_STEP)
    coverage = np.zeros(returns.shape[1], dtype=np.int32)
    for frame in factors.values():
        coverage += np.isfinite(frame.iloc[date_ids].to_numpy()).sum(axis=0)
    coverage += np.isfinite(returns.iloc[date_ids].to_numpy()).sum(axis=0)
    asset_ids = np.argsort(coverage)[-min(MAX_ASSETS, len(coverage)):]
    y = returns.iloc[date_ids, asset_ids].to_numpy(dtype=float)
    panels = []
    for name in names:
        x = factors[name].iloc[date_ids, asset_ids].to_numpy(dtype=float)
        med = np.nanmedian(x, axis=1, keepdims=True)
        lo = np.nanquantile(x, 0.01, axis=1, keepdims=True)
        hi = np.nanquantile(x, 0.99, axis=1, keepdims=True)
        x = np.clip(x, lo, hi) - med
        scale = np.nanmedian(np.abs(x), axis=1, keepdims=True) * 1.4826
        x = np.divide(x, scale, out=np.full_like(x, np.nan), where=scale > 1e-12)
        panels.append(np.clip(x, -8, 8))
    z = np.stack(panels, axis=2)
    return z, y, names, dates[date_ids]


def _daily_betas(z: np.ndarray, y: np.ndarray):
    betas = []
    for t in range(len(y)):
        ok = np.isfinite(y[t]) & np.isfinite(z[t]).all(axis=1)
        if ok.sum() < max(50, z.shape[2] * 5):
            continue
        x = z[t, ok]
        gram = x.T @ x + np.eye(x.shape[1]) * RIDGE
        betas.append(np.linalg.solve(gram, x.T @ y[t, ok]))
    return np.asarray(betas)


def _r2(y: np.ndarray, pred: np.ndarray, baseline=0.0):
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() == 0:
        return np.nan
    sse = np.sum((y[ok] - pred[ok]) ** 2)
    base = np.sum((y[ok] - baseline) ** 2)
    return 1 - sse / base if base > 0 else np.nan


def _ipca_metrics(z: np.ndarray, y: np.ndarray, split: int):
    train_betas = _daily_betas(z[:split], y[:split])
    if train_betas.size == 0:
        return {"oos_predictive_r2": None, "oos_pricing_r2": None}
    mean_beta = np.nanmean(train_betas, axis=0)
    centered = train_betas - mean_beta
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(IPCA_RANK, z.shape[2])].T
    y_all, pred_all, price_all = [], [], []
    for t in range(split, len(y)):
        ok = np.isfinite(y[t]) & np.isfinite(z[t]).all(axis=1)
        if ok.sum() < 50:
            continue
        x, yy = z[t, ok], y[t, ok]
        pred = x @ mean_beta
        managed = x @ basis
        score = np.linalg.lstsq(managed, yy - pred, rcond=None)[0]
        priced = pred + managed @ score
        y_all.append(yy); pred_all.append(pred); price_all.append(priced)
    yy = np.concatenate(y_all); pred = np.concatenate(pred_all); priced = np.concatenate(price_all)
    return {
        "oos_predictive_r2": _finite(_r2(yy, pred, 0.0)),
        "oos_pricing_r2": _finite(_r2(yy, priced, 0.0)),
        "oos_predictive_rmse": _finite(np.sqrt(np.mean((yy - pred) ** 2))),
        "oos_pricing_rmse": _finite(np.sqrt(np.mean((yy - priced) ** 2))),
        "rank": int(basis.shape[1]),
        "interpretation": "预测R²使用训练期平均特征价格；定价R²另允许测试日收益在训练期低秩子空间中实现，后者不是可交易预测。",
    }


def _flatten(z: np.ndarray, y: np.ndarray, start: int, end: int, cap_per_date=350):
    rng = np.random.default_rng(20261026)
    xs, ys = [], []
    for t in range(start, end):
        ok = np.flatnonzero(np.isfinite(y[t]) & np.isfinite(z[t]).all(axis=1))
        if len(ok) > cap_per_date:
            ok = np.sort(rng.choice(ok, cap_per_date, replace=False))
        if len(ok):
            xs.append(z[t, ok]); ys.append(y[t, ok])
    return np.concatenate(xs), np.concatenate(ys)


def _tree_metrics(z: np.ndarray, y: np.ndarray, split: int, names: list[str]):
    xtr, ytr = _flatten(z, y, 0, split)
    xte, yte = _flatten(z, y, split, len(y))

    def best_split(x, target, leaves):
        overall = float(np.nanmean(target))
        base_loss = float(np.sum((target - overall) ** 2))
        best = (base_loss, None, None, None)
        for j in range(x.shape[1]):
            valid = np.isfinite(x[:, j])
            if valid.sum() < 500:
                continue
            thresholds = np.nanquantile(x[valid, j], [.2, .4, .6, .8])
            for threshold in thresholds:
                left = valid & (x[:, j] <= threshold)
                right = valid & (x[:, j] > threshold)
                if left.sum() < 250 or right.sum() < 250:
                    continue
                lm, rm = target[left].mean(), target[right].mean()
                loss = ((target[left] - lm) ** 2).sum() + ((target[right] - rm) ** 2).sum()
                if loss < best[0]:
                    best = (loss, j, float(threshold), (float(lm), float(rm)))
        return best

    loss1, j1, t1, means1 = best_split(xtr, ytr, 2)
    prediction = np.full(len(yte), float(np.mean(ytr)))
    importance = np.zeros(len(names))
    if j1 is not None:
        left_tr = xtr[:, j1] <= t1
        importance[j1] += max(0.0, 1 - loss1 / max(float(np.sum((ytr - ytr.mean()) ** 2)), 1e-12))
        residual = ytr - np.where(left_tr, means1[0], means1[1])
        x2 = xtr.copy()
        # Greedy second global split on residuals within existing leaves.
        split_candidates = []
        for mask in (left_tr, ~left_tr):
            if mask.sum() > 500:
                split_candidates.append((mask, best_split(xtr[mask], residual[mask], 2)))
        leaves = [(j1, t1, means1[0], means1[1], None)]
        for parent_mask, candidate in split_candidates:
            loss2, j2, t2, means2 = candidate
            if j2 is None:
                continue
            parent_ids = np.flatnonzero(parent_mask)
            local_left = xtr[parent_mask, j2] <= t2
            # Store a split under the parent feature/side key.
            parent_left = bool(np.mean(xtr[parent_ids, j1] <= t1) > .5)
            leaves.append((j1, t1, means1[0], means1[1], (parent_left, j2, t2, means2)))
            importance[j2] += max(0.0, (loss1 - loss2) / max(float(np.sum((ytr - ytr.mean()) ** 2)), 1e-12))
        for k in range(len(yte)):
            is_left = np.isfinite(xte[k, j1]) and xte[k, j1] <= t1
            prediction[k] = means1[0] if is_left else means1[1]
            # Apply the first fitted child split on the matching leaf; never refit on test data.
            for _, _, _, _, child in leaves[1:]:
                parent_left, j2, t2, means2 = child
                if bool(is_left) == parent_left and np.isfinite(xte[k, j2]):
                    prediction[k] = means2[0] if xte[k, j2] <= t2 else means2[1]
                    break
    return {
        "oos_predictive_r2": _finite(_r2(yte, prediction, 0.0)),
        "oos_predictive_rmse": _finite(np.sqrt(np.mean((yte - prediction) ** 2))),
        "feature_importance": [{"factor": names[i], "importance": float(v)}
                               for i, v in sorted(enumerate(importance), key=lambda item: -item[1])],
        "depth": 2, "split_feature": names[j1] if j1 is not None else None,
        "interpretation": "训练期估计并冻结的两层全局切分树；预测期不重估，不是完整P-Tree/SDF估计。",
    }


def run_model_diagnostics(factors: dict[str, pd.DataFrame], forward: dict[str, pd.DataFrame]):
    z, y, names, sample_dates = _sample_panel(factors, forward["1d"])
    split = max(20, int(len(y) * TRAIN_FRACTION))
    ipca = _ipca_metrics(z, y, split)
    tree = _tree_metrics(z, y, split, names)
    ablation = []
    for i, name in enumerate(names):
        keep = np.arange(len(names)) != i
        ipca_without = _ipca_metrics(z[:, :, keep], y, split)
        tree_without = _tree_metrics(z[:, :, keep], y, split, [n for j, n in enumerate(names) if j != i])
        ablation.append({
            "factor": name,
            "ipca_delta_predictive_r2": _finite((ipca.get("oos_predictive_r2") or 0) - (ipca_without.get("oos_predictive_r2") or 0)),
            "ipca_delta_pricing_r2": _finite((ipca.get("oos_pricing_r2") or 0) - (ipca_without.get("oos_pricing_r2") or 0)),
            "tree_delta_predictive_r2": _finite((tree.get("oos_predictive_r2") or 0) - (tree_without.get("oos_predictive_r2") or 0)),
        })
    return {
        "protocol": {
            "label": "探索性模型诊断", "train_fraction": TRAIN_FRACTION,
            "sampled_dates": int(len(y)), "sampled_assets": int(z.shape[1]),
            "date_step": DATE_STEP, "test_start": str(pd.Timestamp(sample_dates[split]).date()),
            "warning": "IPCA/P-Tree-inspired educational approximations; do not interpret as paper replication or executable alpha.",
        },
        "ipca": ipca, "pricing_tree": tree, "ablation": ablation,
    }
