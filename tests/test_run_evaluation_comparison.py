"""Tests for the Batch-2 additions to scripts/run_evaluation.py (replay
builder, PS-04 burden, output writer). Uses small hand-built frames only."""
import numpy as np
import pandas as pd
import pytest

import scripts.run_evaluation as RE
from src.scoring import ALERT, NO_ALERT, REVIEW

D0 = pd.Timestamp("2026-05-10")


def day(n):
    return D0 + pd.Timedelta(days=n)


def _full(days, metric="spend", cid="A"):
    return pd.DataFrame([{
        "date": day(d), "campaign_id": cid, "metric": metric, "observed": 100.0 + d,
        "baseline": 100.0, "z": 5.0 if rt != NO_ALERT else 0.0, "severity": "high",
        "evidence_score": 0.7, "route": rt, "notification": note, "episode_id": ep,
        "period": "eval", "explanation": "x",
    } for d, rt, note, ep in days])


def _labels():
    return pd.DataFrame([{
        "incident_id": "I1", "campaign_id": "A", "incident_type": "spend_spike",
        "start_date": day(0), "end_date": day(2), "affected_metric": "spend",
        "severity": "high", "magnitude": 2.0, "period": "eval",
    }, {
        "incident_id": "I0", "campaign_id": "A", "incident_type": "spend_spike",
        "start_date": day(-60), "end_date": day(-58), "affected_metric": "spend",
        "severity": "high", "magnitude": 2.0, "period": "dev",
    }])


def test_replay_window_padding_fixed_join_and_eval_only():
    days = [(d, ALERT if d in (0, 1) else NO_ALERT, {0: "new", 1: "suppressed"}.get(d, ""),
             "E1" if d in (0, 1) else pd.NA) for d in range(-10, 13)]
    full = _full(days)
    full = pd.concat([full, _full([(0, ALERT, "new", "E9")], metric="ctr")])  # other metric
    fixed = full[full["metric"] == "spend"][["campaign_id", "date", "metric"]].assign(
        route=NO_ALERT, notification="", pct_change=0.0)
    fixed.loc[fixed["date"] == day(1), ["route", "notification"]] = [ALERT, "new"]

    rep = RE.build_replay(full, fixed, _labels(), pad=7)
    assert set(rep["incident_id"]) == {"I1"}  # DEV incident excluded
    assert rep["date"].min() == day(-7) and rep["date"].max() == day(9)
    assert set(rep["metric"]) == {"spend"}
    assert rep["in_incident_window"].sum() == 3
    assert rep.loc[rep["date"] == day(1), "fixed_notification"].item() == "new"
    assert rep.loc[rep["date"] == day(0), "notification"].item() == "new"


def test_ps04_burden_records_failure_instead_of_hiding_it(monkeypatch):
    def boom(*a, **k):
        raise ValueError("unsupported column")
    monkeypatch.setattr(RE, "fixed_threshold_records", boom)
    recs = _full([(0, ALERT, "new", "E1"), (1, REVIEW, "suppressed", "E1"),
                  (2, NO_ALERT, "", pd.NA)])
    # Empty PS-04 frame: the fixed baseline cannot run. It must be recorded as
    # not_run with the exception text, and the other two systems still reported.
    df = RE.run_ps04_burden(pd.DataFrame(), recs, 0.5).set_index("system")
    assert df.loc["fixed_threshold", "status"] == "not_run"
    assert "Error" in df.loc["fixed_threshold", "note"]
    assert np.isnan(df.loc["fixed_threshold", "notifications_sent"])
    assert df.loc["full_adomaly", "notifications_sent"] == 1
    assert df.loc["adaptive_detector", "notifications_sent"] == 2
    assert (df["dataset"] == "PS-04 (official, no labels)").all()


def test_comparison_writer_appends_and_keeps_existing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(RE, "REPORT_DIR", tmp_path)
    (tmp_path / "report.md").write_text("# Existing sections 1-7\n", encoding="utf-8")
    comp = pd.DataFrame([{
        "system": k, "system_label": RE.SYSTEM_LABELS[k], "precision": 0.5, "recall": 0.5,
        "f1": 0.5, "false_positive_rate": 0.1, "incidents_detected": 1, "incidents_notified": 1,
        "incidents_missed": 0, "n_incidents": 1, "incidents_detected_not_notified": 0,
        "incidents_reaching_human_via_review_only": 0, "median_notification_delay_days": np.nan,
        "raw_flagged_records": 4, "notifications_sent": 2, "notification_reduction_pct": 50.0,
    } for k in RE.SYSTEMS])
    cmp = {"threshold": 0.5, "grid": pd.DataFrame({"threshold": [0.5], "f1": [0.5]}),
           "comparison": comp, "coverage": pd.DataFrame({"system": ["full_adomaly"]}),
           "route_precision": {"full_adomaly": {ALERT: {"flags": 1, "tp": 1, "fp": 0,
                                                       "precision": np.float64(1.0)}}},
           "route_precision_df": pd.DataFrame([{"system": "full_adomaly",
                                                "system_label": "Full", "route": ALERT,
                                                "flags": 1, "tp": 1, "fp": 0, "precision": 1.0}]),
           "replay": pd.DataFrame({"incident_id": ["I1"]})}
    burden = pd.DataFrame([{"dataset": "PS-04 (official, no labels)", "system": k,
                            "system_label": RE.SYSTEM_LABELS[k], "raw_flagged_records": 4,
                            "notifications_sent": 2, "suppressed_repeats": 2,
                            "notification_reduction_pct": 50.0, "status": "ok", "note": ""}
                           for k in RE.SYSTEMS])
    RE.write_comparison_outputs(cmp, burden)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# Existing sections 1-7\n")
    assert "## 8. Controlled comparison" in text and "## 9. PS-04" in text
    for f in ["system_comparison.csv", "system_comparison.json", "fixed_threshold_grid.csv",
              "incident_notification_coverage.csv", "route_precision.csv",
              "synthetic_incident_replay.csv", "ps04_notification_burden.csv"]:
        assert (tmp_path / f).exists(), f
    assert list(pd.read_csv(tmp_path / "system_comparison.csv")["system"]) == list(RE.SYSTEMS)
