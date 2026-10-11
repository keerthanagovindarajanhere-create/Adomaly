"""Tests for the Round-2 additions: src/baselines.py and the comparison
functions appended to src/evaluation.py. Existing tests in
tests/test_evaluation.py are unchanged and cover the original functions."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.baselines import (DOD_GRID, fixed_threshold_records, pick_threshold,
                           select_fixed_threshold)
from src.data_io import load_benign, load_labels, load_metrics
from src.evaluation import (NOTIFYING, notification_burden, notification_coverage,
                            notify_every_flag, route_precision, system_metrics)
from src.pipeline import run_detection
from src.scoring import ALERT, NO_ALERT, REVIEW

D0 = pd.Timestamp("2026-04-10")


def day(n):
    return D0 + pd.Timedelta(days=n)


def make_metrics(n=10, cid="T01", spend=100.0):
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n), "campaign_id": cid,
        "spend": spend, "impressions": 10_000, "clicks": 200, "conversions": 10,
    })


# ======================================================== fixed baseline ==
def test_grid_is_20_to_90_percent():
    assert DOD_GRID == (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _spend(rec, d):
    return rec[(rec["metric"] == "spend") & (rec["date"] == pd.Timestamp("2026-01-01") + pd.Timedelta(days=d))].iloc[0]


def test_threshold_is_inclusive_and_both_directions():
    m = make_metrics()
    m.loc[3, "spend"] = 150.0   # +50% vs day 2
    m.loc[4, "spend"] = 75.0    # -50% vs day 3
    rec = fixed_threshold_records(m, 0.5)
    assert _spend(rec, 3)["route"] == ALERT and _spend(rec, 3)["direction"] == "up"
    assert _spend(rec, 4)["route"] == ALERT and _spend(rec, 4)["direction"] == "down"
    assert _spend(rec, 2)["route"] == NO_ALERT
    assert (rec.loc[rec["route"] == ALERT, "notification"] == "new").all()
    assert (rec.loc[rec["route"] == NO_ALERT, "notification"] == "").all()


def test_first_day_zero_and_missing_previous_are_not_scored():
    m = make_metrics()
    m["spend"] = m["spend"].astype(float)
    m.loc[3, "spend"] = 0.0      # day 4 has zero previous value
    m.loc[6, "spend"] = np.nan   # day 7 has missing previous; day 6 itself missing
    rec = fixed_threshold_records(m, 0.2)
    assert _spend(rec, 0)["status"] == "no_previous"
    assert _spend(rec, 4)["status"] == "no_previous" and _spend(rec, 4)["route"] == NO_ALERT
    assert _spend(rec, 6)["status"] == "missing_value" and _spend(rec, 6)["route"] == NO_ALERT
    assert _spend(rec, 7)["status"] == "no_previous" and _spend(rec, 7)["route"] == NO_ALERT
    # the 100 -> 0 drop on day 3 is a valid -100% change and is flagged
    assert _spend(rec, 3)["route"] == ALERT


def test_baseline_uses_previous_day_only():
    m = make_metrics(n=12)
    m2 = m.copy()
    m2.loc[8:, ["spend", "impressions", "clicks", "conversions"]] *= 5
    a = fixed_threshold_records(m, 0.3)
    b = fixed_threshold_records(m2, 0.3)
    cut = pd.Timestamp("2026-01-08")  # day 7, last day before the change
    pd.testing.assert_frame_equal(a[a["date"] <= cut].reset_index(drop=True),
                                  b[b["date"] <= cut].reset_index(drop=True))


def test_baseline_scores_same_metrics_as_detector_including_ratios():
    rec = fixed_threshold_records(make_metrics(), 0.5)
    assert set(rec["metric"]) == {"spend", "impressions", "clicks", "conversions", "ctr", "cvr"}
    ctr = rec[(rec["metric"] == "ctr")]["observed"].dropna()
    assert np.allclose(ctr, 200 / 10_000)


def test_baseline_rejects_label_columns():
    with pytest.raises(ValueError):
        fixed_threshold_records(make_metrics().assign(incident_type="x"), 0.5)


def test_pick_threshold_best_f1_ties_go_to_largest_nan_last():
    g = pd.DataFrame({"threshold": [0.2, 0.3, 0.4, 0.5], "f1": [0.1, 0.3, np.nan, 0.3]})
    assert pick_threshold(g) == 0.5
    g = pd.DataFrame({"threshold": [0.2, 0.3], "f1": [np.nan, np.nan]})
    assert pick_threshold(g) == 0.3


@pytest.fixture(scope="module")
def bench_data():
    return load_metrics(), load_labels(), load_benign()


def test_threshold_selection_ignores_eval_data(bench_data):
    m, labels, benign = bench_data
    th1, g1 = select_fixed_threshold(m, labels, benign)

    m2 = m.copy()
    ev = m2["date"] > C.DEV_END
    m2.loc[ev, ["spend", "impressions", "clicks", "conversions"]] *= 3
    labels2 = labels.copy()
    labels2.loc[labels2["period"] == "eval", "campaign_id"] = "C01"  # scramble eval labels
    th2, g2 = select_fixed_threshold(m2, labels2, benign)

    assert th1 == th2
    pd.testing.assert_frame_equal(g1, g2)
    assert set(g1["threshold"]) == set(DOD_GRID)
    assert th1 in DOD_GRID


# ================================================== notification coverage ==
def _rec(d, route, note, metric="spend", cid="T01", ep="EP1", period="eval"):
    return {"campaign_id": cid, "date": day(d), "metric": metric, "route": route,
            "notification": note, "episode_id": ep, "period": period}


def _labels(*incs):
    return pd.DataFrame([{
        "incident_id": iid, "campaign_id": cid, "incident_type": "spend_spike",
        "start_date": day(s), "end_date": day(e), "affected_metric": metric,
        "severity": "high", "magnitude": 2.0, "period": "eval",
    } for iid, cid, s, e, metric in incs])


def test_coverage_three_statuses_and_suppression_from_earlier_episode():
    recs = pd.DataFrame([
        # I1: notified on day 0 of its window
        _rec(0, ALERT, "new", cid="A", ep="E1"),
        _rec(1, ALERT, "suppressed", cid="A", ep="E1"),
        # I2: episode began the day BEFORE the window -> in-window flags suppressed
        _rec(4, ALERT, "new", cid="B", ep="E2"),
        _rec(5, ALERT, "suppressed", cid="B", ep="E2"),
        _rec(6, ALERT, "suppressed", cid="B", ep="E2"),
        # I3: nothing flagged inside its window
        _rec(0, NO_ALERT, "", cid="C", ep=pd.NA),
    ])
    labels = _labels(("I1", "A", 0, 2, "spend"), ("I2", "B", 5, 7, "spend"),
                     ("I3", "C", 0, 3, "spend"))
    cov = notification_coverage(recs, labels).set_index("incident_id")

    assert cov.loc["I1", "status"] == "notified" and cov.loc["I1", "notification_delay_days"] == 0
    assert cov.loc["I2", "status"] == "detected_not_notified"
    assert cov.loc["I2", "detected"] and not cov.loc["I2", "notified"]
    assert cov.loc["I2", "suppressed_in_window"] == 2
    assert cov.loc["I2", "episode_started_before_window"]
    assert np.isnan(cov.loc["I2", "notification_delay_days"])
    assert cov.loc["I3", "status"] == "missed" and not cov.loc["I3", "detected"]


def test_escalation_in_window_counts_as_notified_and_fragmentation_counted():
    recs = pd.DataFrame([
        _rec(-1, REVIEW, "new", ep="E1"),
        _rec(0, ALERT, "escalation", ep="E1"),      # escalation inside window
        _rec(3, ALERT, "new", ep="E2"),             # second episode, same incident
    ])
    cov = notification_coverage(recs, _labels(("I1", "T01", 0, 4, "spend"))).iloc[0]
    assert cov["status"] == "notified"
    assert cov["notifications_in_window"] == 2
    assert cov["episode_started_before_window"]
    assert not cov["review_only"]


def test_notification_on_other_metric_does_not_count():
    recs = pd.DataFrame([_rec(1, ALERT, "new", metric="cvr")])
    labels = _labels(("I1", "T01", 0, 3, "conversion_rate"))  # mapped to conversions
    cov = notification_coverage(recs, labels).iloc[0]
    assert cov["mapped_metric"] == "conversions" and cov["status"] == "missed"


def test_review_only_flag():
    recs = pd.DataFrame([_rec(1, REVIEW, "new"), _rec(2, REVIEW, "suppressed")])
    cov = notification_coverage(recs, _labels(("I1", "T01", 0, 3, "spend"))).iloc[0]
    assert cov["review_only"] and cov["status"] == "notified"


def test_coverage_without_episode_column_is_safe():
    recs = pd.DataFrame([_rec(1, ALERT, "new")]).drop(columns="episode_id")
    cov = notification_coverage(recs, _labels(("I1", "T01", 0, 3, "spend"))).iloc[0]
    assert cov["status"] == "notified" and not cov["episode_started_before_window"]


# ====================================================== system variants ==
def test_notify_every_flag_keeps_flags_and_notifies_each():
    recs = pd.DataFrame([_rec(0, REVIEW, "new"), _rec(1, ALERT, "suppressed"),
                         _rec(2, NO_ALERT, "", ep=pd.NA)])
    out = notify_every_flag(recs)
    assert list(out["route"]) == [ALERT, ALERT, NO_ALERT]
    assert list(out["notification"]) == ["new", "new", ""]
    assert out["episode_id"].isna().all()
    assert list(recs["route"]) == [REVIEW, ALERT, NO_ALERT]  # input not mutated


def test_route_precision_and_burden():
    lab = pd.DataFrame({
        "metric": ["spend", "spend", "spend", "cvr"],
        "route": [ALERT, ALERT, REVIEW, ALERT],
        "is_incident_positive": [True, False, True, False],
        "notification": ["new", "suppressed", "new", "new"],
    })
    rp = route_precision(lab)
    assert rp[ALERT] == {"flags": 2, "tp": 1, "fp": 1, "precision": 0.5}  # cvr excluded
    assert rp[REVIEW]["precision"] == 1.0
    b = notification_burden(lab)  # all metrics, including cvr
    assert b["raw_flagged_records"] == 4 and b["notifications_sent"] == 3
    assert b["notification_reduction_pct"] == pytest.approx(25.0)


def test_burden_with_no_flags_is_nan_not_crash():
    b = notification_burden(pd.DataFrame({"route": [NO_ALERT], "notification": [""]}))
    assert b["raw_flagged_records"] == 0 and np.isnan(b["notification_reduction_pct"])


# ================================== end-to-end on the synthetic benchmark ==
@pytest.fixture(scope="module")
def systems(bench_data):
    m, labels, benign = bench_data
    full, _ = run_detection(m)
    th, _ = select_fixed_threshold(m, labels, benign)
    return {
        "fixed": system_metrics("fixed", fixed_threshold_records(m, th), labels, benign),
        "adaptive": system_metrics("adaptive", notify_every_flag(full), labels, benign),
        "full": system_metrics("full", full, labels, benign),
    }


def test_adaptive_and_full_share_detection_quality(systems):
    a, f = systems["adaptive"], systems["full"]
    for k in ("tp", "fp", "fn", "tn", "raw_flagged_records", "incidents_detected"):
        assert a[k] == f[k], k
    assert a["notifications_sent"] == a["raw_flagged_records"]
    assert f["notifications_sent"] <= a["notifications_sent"]


def test_system_metrics_internally_consistent(systems):
    for s in systems.values():
        assert s["period"] == "eval"
        assert s["n_incidents"] == (s["incidents_notified"] + s["incidents_detected_not_notified"]
                                    + s["incidents_missed"])
        assert s["incidents_notified"] <= s["incidents_detected"] <= s["n_incidents"]
        assert s["true_incident_notifications"] + s["false_notifications"] == \
            s["notifications_on_labelled_metrics"]
        if s["fp"] + s["tn"]:
            assert s["false_positive_rate"] == pytest.approx(s["fp"] / (s["fp"] + s["tn"]))
    assert systems["fixed"]["incidents_reaching_human_via_review_only"] == 0
    assert systems["adaptive"]["incidents_reaching_human_via_review_only"] == 0


def test_notifying_values():
    assert NOTIFYING == ("new", "escalation")
