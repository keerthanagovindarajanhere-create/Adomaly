"""Targeted tests for src/evaluation.py: matching rule, record-level P/R/F1,
incident-level detection/delay, benign false-alert counting, and alert-fatigue
summarization. Uses small hand-built record/label frames, not the full
benchmark, so each case is independently checkable.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    MAPPED_METRICS,
    METRIC_MAP,
    alert_fatigue_summary,
    false_alerts_on_benign,
    incident_level_results,
    label_records,
    record_level_metrics,
)
from src.scoring import ALERT, NO_ALERT, REVIEW


def _records(rows):
    """rows: (campaign_id, metric, date, route). period fixed to 'eval'."""
    return pd.DataFrame([{
        "campaign_id": c, "metric": m, "date": pd.Timestamp(d), "route": rt,
        "period": "eval",
    } for c, m, d, rt in rows])


def _labels(rows):
    """rows: (incident_id, campaign_id, affected_metric, start, end, severity)."""
    return pd.DataFrame([{
        "incident_id": i, "campaign_id": c, "incident_type": "spend_spike",
        "start_date": pd.Timestamp(s), "end_date": pd.Timestamp(e),
        "affected_metric": m, "severity": sev, "magnitude": 1.0, "period": "eval",
    } for i, c, m, s, e, sev in rows])


def _benign(rows):
    """rows: (event_id, campaign_id, affected_metric, start)."""
    return pd.DataFrame([{
        "event_id": i, "campaign_id": c, "event_type": "seasonal_budget_increase",
        "affected_metric": m, "start_date": pd.Timestamp(s), "magnitude": 1.0,
        "period": "eval",
    } for i, c, m, s in rows])


EMPTY_BENIGN = _benign([])


# ---------------------------------------------------------------- labeling --
def test_metric_map_covers_all_affected_metric_values():
    # affected_metric values that appear in src/data_generation.py INCIDENTS
    for m in ("spend", "conversions", "ctr", "conversion_rate"):
        assert m in METRIC_MAP
    assert METRIC_MAP["conversion_rate"] == "conversions"
    assert set(MAPPED_METRICS) == {"spend", "conversions", "ctr"}


def test_label_records_marks_incident_window_inclusive():
    records = _records([
        ("C01", "spend", "2026-01-01", NO_ALERT),
        ("C01", "spend", "2026-01-02", ALERT),   # inside window, start
        ("C01", "spend", "2026-01-03", ALERT),   # inside window, end
        ("C01", "spend", "2026-01-04", NO_ALERT),  # just after window
    ])
    labels = _labels([("INC1", "C01", "spend", "2026-01-02", "2026-01-03", "high")])
    out = label_records(records, labels, EMPTY_BENIGN)
    flags = out.set_index("date")["is_incident_positive"]
    assert flags[pd.Timestamp("2026-01-01")] == False
    assert flags[pd.Timestamp("2026-01-02")] == True
    assert flags[pd.Timestamp("2026-01-03")] == True
    assert flags[pd.Timestamp("2026-01-04")] == False


def test_label_records_conversion_rate_maps_to_conversions_metric():
    records = _records([("C01", "conversions", "2026-01-02", ALERT),
                        ("C01", "cvr", "2026-01-02", ALERT)])
    labels = _labels([("INC1", "C01", "conversion_rate", "2026-01-01", "2026-01-05", "low")])
    out = label_records(records, labels, EMPTY_BENIGN)
    conv_row = out[out["metric"] == "conversions"].iloc[0]
    cvr_row = out[out["metric"] == "cvr"].iloc[0]
    assert conv_row["is_incident_positive"] == True
    assert cvr_row["is_incident_positive"] == False  # cvr is not a mapped metric


def test_label_records_benign_shift_from_start_onward_no_end():
    records = _records([
        ("C06", "spend", "2026-01-01", NO_ALERT),  # before shift
        ("C06", "spend", "2026-01-05", ALERT),      # on/after shift start, no end date
        ("C06", "spend", "2026-06-01", ALERT),      # far after, still "shifted"
    ])
    benign = _benign([("BEN1", "C06", "spend", "2026-01-05")])
    out = label_records(records, pd.DataFrame(columns=[
        "incident_id", "campaign_id", "incident_type", "start_date", "end_date",
        "affected_metric", "severity", "magnitude", "period"]), benign)
    shift = out.set_index("date")["is_benign_shift"]
    assert shift[pd.Timestamp("2026-01-01")] == False
    assert shift[pd.Timestamp("2026-01-05")] == True
    assert shift[pd.Timestamp("2026-06-01")] == True


# ------------------------------------------------------- record-level P/R/F1
def test_record_level_metrics_perfect_detection():
    records = _records([
        ("C01", "spend", "2026-01-02", ALERT),
        ("C01", "spend", "2026-01-03", ALERT),
        ("C02", "spend", "2026-01-02", NO_ALERT),
    ])
    labels = _labels([("INC1", "C01", "spend", "2026-01-02", "2026-01-03", "high")])
    labelled = label_records(records, labels, EMPTY_BENIGN)
    r = record_level_metrics(labelled)
    assert r.tp == 2 and r.fp == 0 and r.fn == 0
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0


def test_record_level_metrics_counts_fp_and_fn():
    records = _records([
        ("C01", "spend", "2026-01-02", ALERT),    # TP
        ("C01", "spend", "2026-01-03", NO_ALERT),  # FN (missed day in window)
        ("C02", "spend", "2026-01-10", ALERT),    # FP (not in any window)
    ])
    labels = _labels([("INC1", "C01", "spend", "2026-01-02", "2026-01-03", "high")])
    labelled = label_records(records, labels, EMPTY_BENIGN)
    r = record_level_metrics(labelled)
    assert r.tp == 1 and r.fp == 1 and r.fn == 1
    assert r.precision == 0.5 and r.recall == 0.5
    assert math.isclose(r.f1, 0.5, rel_tol=1e-9)


def test_record_level_metrics_excludes_unmapped_metrics_from_scoring():
    records = _records([
        ("C01", "cvr", "2026-01-02", ALERT),          # unmapped: not scored as FP
        ("C01", "impressions", "2026-01-02", ALERT),  # unmapped: not scored as FP
        ("C01", "spend", "2026-01-02", NO_ALERT),
    ])
    labels = _labels([])
    labelled = label_records(records, labels, EMPTY_BENIGN)
    r = record_level_metrics(labelled)
    assert r.n_evaluated == 1  # only the "spend" row is in the mapped-metric universe
    assert r.fp == 0
    assert r.n_unmapped_metric_flags == 2


def test_record_level_metrics_no_positives_gives_nan_recall_not_crash():
    records = _records([("C01", "spend", "2026-01-02", NO_ALERT)])
    labelled = label_records(records, _labels([]), EMPTY_BENIGN)
    r = record_level_metrics(labelled)
    assert r.tp == 0 and r.fn == 0
    assert math.isnan(r.recall)  # 0/0 undefined, not silently 0 or 1


# ------------------------------------------------------ incident-level view
def test_incident_level_detected_true_with_correct_delay():
    records = _records([
        ("C01", "spend", "2026-01-01", NO_ALERT),
        ("C01", "spend", "2026-01-02", NO_ALERT),
        ("C01", "spend", "2026-01-03", ALERT),  # first flag, 1 day after start
        ("C01", "spend", "2026-01-04", ALERT),
    ])
    labels = _labels([("INC1", "C01", "spend", "2026-01-02", "2026-01-04", "high")])
    out = incident_level_results(records, labels)
    row = out.iloc[0]
    assert row["detected"] == True
    assert row["delay_days"] == 1
    assert row["first_flag_date"] == pd.Timestamp("2026-01-03")


def test_incident_level_not_detected_gives_nan_delay():
    records = _records([("C01", "spend", "2026-01-02", NO_ALERT),
                        ("C01", "spend", "2026-01-03", NO_ALERT)])
    labels = _labels([("INC1", "C01", "spend", "2026-01-02", "2026-01-03", "high")])
    out = incident_level_results(records, labels)
    row = out.iloc[0]
    assert row["detected"] == False
    assert math.isnan(row["delay_days"])


def test_incident_level_human_review_counts_as_detected():
    records = _records([("C01", "spend", "2026-01-02", REVIEW)])
    labels = _labels([("INC1", "C01", "spend", "2026-01-02", "2026-01-02", "low")])
    out = incident_level_results(records, labels)
    assert out.iloc[0]["detected"] == True
    assert out.iloc[0]["best_route"] == REVIEW


# ------------------------------------------------------------- benign false alerts
def test_false_alerts_on_benign_counts_flags_only():
    records = _records([
        ("C06", "spend", "2026-01-05", ALERT),
        ("C06", "spend", "2026-01-06", REVIEW),
        ("C06", "spend", "2026-01-07", NO_ALERT),
    ])
    benign = _benign([("BEN1", "C06", "spend", "2026-01-05")])
    labelled = label_records(records, _labels([]), benign)
    out = false_alerts_on_benign(labelled)
    assert out["benign_shift_records"] == 3
    assert out["benign_shift_flagged_records"] == 2
    assert out["benign_shift_alert_records"] == 1
    assert out["benign_events_with_any_flag"] == ["BEN1"]


def test_false_alerts_on_benign_empty_is_safe():
    records = _records([("C01", "spend", "2026-01-01", NO_ALERT)])
    labelled = label_records(records, _labels([]), EMPTY_BENIGN)
    out = false_alerts_on_benign(labelled)
    assert out["benign_shift_records"] == 0
    assert math.isnan(out["benign_shift_flag_rate"])


# -------------------------------------------------------------- alert fatigue
def test_alert_fatigue_summary_matches_episode_counts():
    records = pd.DataFrame([
        {"route": ALERT, "notification": "new"},
        {"route": ALERT, "notification": "suppressed"},
        {"route": REVIEW, "notification": "escalation"},
        {"route": NO_ALERT, "notification": ""},
    ])
    episodes = pd.DataFrame({"episode_id": ["EP1"]})
    out = alert_fatigue_summary(records, episodes)
    assert out["raw_flagged_records"] == 3
    assert out["alerts"] == 2 and out["human_review_records"] == 1
    assert out["notifications_sent"] == 2  # new + escalation
    assert out["suppressed_repeats"] == 1
    assert out["episodes"] == 1
    assert math.isclose(out["notification_reduction_pct"], (3 - 2) / 3 * 100)


def test_alert_fatigue_summary_no_flags_is_safe():
    records = pd.DataFrame({"route": [NO_ALERT, NO_ALERT], "notification": ["", ""]})
    episodes = pd.DataFrame({"episode_id": []})
    out = alert_fatigue_summary(records, episodes)
    assert out["raw_flagged_records"] == 0
    assert math.isnan(out["notification_reduction_pct"])


# ------------------------------------------------- integration with real pipeline
def test_evaluation_runs_end_to_end_on_real_synthetic_benchmark():
    """Smoke test against the actual benchmark data/pipeline (not hand-built
    frames): catches schema drift between src/evaluation.py and src/pipeline.py."""
    from src.data_io import load_benign, load_labels, load_metrics
    from src.pipeline import run_detection

    metrics, labels, benign = load_metrics(), load_labels(), load_benign()
    records, episodes = run_detection(metrics)
    labelled = label_records(records, labels, benign)
    eval_labelled = labelled[labelled["period"] == "eval"]

    r = record_level_metrics(eval_labelled)
    assert 0 <= r.precision <= 1 or math.isnan(r.precision)
    assert 0 <= r.recall <= 1 or math.isnan(r.recall)
    assert r.tp + r.fn == r.n_positive
    assert r.tp + r.fp + r.fn + r.tn == r.n_evaluated

    inc = incident_level_results(records, labels)
    assert set(inc["incident_id"]) == set(labels["incident_id"])
    # every incident with a delay must be marked detected, and vice versa
    assert (inc["detected"] == inc["delay_days"].notna()).all()

    fatigue = alert_fatigue_summary(records, episodes)
    assert fatigue["notifications_sent"] <= fatigue["raw_flagged_records"]
