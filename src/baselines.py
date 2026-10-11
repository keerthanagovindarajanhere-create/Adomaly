"""Fixed-threshold comparison baseline for the Adomaly evaluation.

This is the "rule-based alerting" reference system: flag a campaign-day-metric
whenever it moved by at least a fixed percentage versus the previous day.
It has no campaign-specific baseline, no weekday adjustment, no evidence
routing and no episode grouping, so every flag is a notification.

Rule (documented, applied identically to every campaign and metric)
------------------------------------------------------------------
    pct_change(t) = (value(t) - value(t-1)) / value(t-1)
    flag  <=>  |pct_change(t)| >= threshold        (inclusive)

- Metrics: the same set the adaptive detector scores (cfg.metrics), with CTR
  and CVR built exactly as the detector builds them
  (src.detection.prepare_series), so notification burden is comparable.
- Only day t-1 is used to score day t, so no future data can change a score.
- Zero or missing previous-day value: the percentage change is undefined, so
  the record is NOT scored (status "no_previous") and never flagged. A missing
  value on day t itself gives status "missing_value", also never flagged.
  The first day of each campaign has no previous day and is "no_previous".

Threshold selection (select_fixed_threshold)
--------------------------------------------
- Grid: 20%, 30%, ..., 90% (DOD_GRID).
- Criterion: record-level F1 on the DEV period only, computed with the same
  labelled universe and functions as the main evaluation
  (src.evaluation.label_records / record_level_metrics).
- Isolation: metrics after C.DEV_END and labels/benign events outside the DEV
  period are removed BEFORE anything is computed, so EVAL data cannot
  influence the choice.
- Tie-break: if several thresholds share the best F1 (to 1e-12), the LARGEST
  threshold wins, i.e. the more conservative rule that sends fewer
  notifications. A NaN F1 (no flags or no positives) ranks below any number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data_io import assign_period
from src.detection import prepare_series
from src.detector_config import DetectorConfig
from src.scoring import ALERT, NO_ALERT

DOD_GRID = tuple(round(x / 10, 1) for x in range(2, 10))  # 0.2 .. 0.9

BASELINE_COLUMNS = [
    "campaign_id", "date", "period", "metric", "observed", "previous", "pct_change",
    "direction", "status", "route", "notification",
]


def fixed_threshold_records(metrics: pd.DataFrame, threshold: float,
                            cfg: DetectorConfig | None = None,
                            allowed_columns: list[str] | None = None) -> pd.DataFrame:
    """Score `metrics` with the day-over-day rule. Returns one row per
    campaign x date x metric with route Alert/No Alert and notification
    "new" for every flag ("" otherwise)."""
    if not 0 < threshold:
        raise ValueError("threshold must be positive (a fraction, e.g. 0.5 for 50%)")
    cfg = cfg or DetectorConfig()
    allowed = allowed_columns or C.METRIC_COLUMNS
    extra = set(metrics.columns) - set(allowed)
    if extra:  # same anti-leakage guard as src.pipeline.run_detection
        raise ValueError(f"Baseline received non-metric columns: {extra}")

    wide = prepare_series(metrics)
    frames = []
    for cid, g in wide.groupby("campaign_id", sort=True):
        g = g.sort_values("date")
        for metric in cfg.metrics:
            obs = g[metric].to_numpy(dtype=float)
            prev = np.concatenate([[np.nan], obs[:-1]])
            ok_prev = ~np.isnan(prev) & (prev > 0)
            ok_obs = ~np.isnan(obs)
            scored = ok_prev & ok_obs
            pct = np.full(len(obs), np.nan)
            pct[scored] = (obs[scored] - prev[scored]) / prev[scored]
            status = np.where(~ok_obs, "missing_value",
                              np.where(~ok_prev, "no_previous", "scored"))
            flag = scored & (np.abs(np.nan_to_num(pct)) >= threshold - 1e-12)
            frames.append(pd.DataFrame({
                "campaign_id": cid, "date": g["date"].to_numpy(), "metric": metric,
                "observed": obs, "previous": prev, "pct_change": pct,
                "direction": np.where(pct > 0, "up", np.where(pct < 0, "down", "flat")),
                "status": status,
                "route": np.where(flag, ALERT, NO_ALERT),
                "notification": np.where(flag, "new", ""),
            }))
    out = pd.concat(frames, ignore_index=True)
    out["period"] = assign_period(out["date"])
    return out[BASELINE_COLUMNS]


def pick_threshold(grid: pd.DataFrame) -> float:
    """Best F1; ties -> largest threshold; NaN F1 ranks last."""
    f1 = grid["f1"].fillna(-np.inf)
    best = f1.max()
    tied = grid.loc[np.isclose(f1, best, rtol=0, atol=1e-12) | (f1 == best), "threshold"]
    return float(tied.max())


def select_fixed_threshold(metrics: pd.DataFrame, labels: pd.DataFrame,
                           benign: pd.DataFrame, grid=DOD_GRID,
                           cfg: DetectorConfig | None = None):
    """Choose the day-over-day threshold on DEV data only.

    Returns (threshold, grid_results) where grid_results has one row per
    candidate with its DEV-period TP/FP/FN/TN, precision, recall, F1 and
    number of flags.
    """
    # Local import: evaluation imports nothing from here, but keep the
    # dependency one-way and explicit.
    from src.evaluation import label_records, record_level_metrics

    dev_metrics = metrics[pd.to_datetime(metrics["date"]) <= C.DEV_END]
    dev_labels = labels[labels["period"] == "dev"]
    dev_benign = benign[pd.to_datetime(benign["start_date"]) <= C.DEV_END]

    rows = []
    for th in grid:
        rec = fixed_threshold_records(dev_metrics, th, cfg)
        lab = label_records(rec, dev_labels, dev_benign)
        rl = record_level_metrics(lab[lab["period"] == "dev"])
        rows.append({"threshold": float(th), "tp": rl.tp, "fp": rl.fp, "fn": rl.fn,
                     "tn": rl.tn, "precision": rl.precision, "recall": rl.recall,
                     "f1": rl.f1, "n_flagged_all_metrics": int((rec["route"] == ALERT).sum())})
    grid_df = pd.DataFrame(rows)
    return pick_threshold(grid_df), grid_df
