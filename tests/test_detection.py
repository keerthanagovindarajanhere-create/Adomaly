import inspect

import numpy as np
import pandas as pd
import pytest

from src import pipeline
from src.data_io import load_labels, load_metrics
from src.detection import compute_baselines
from src.detector_config import DetectorConfig
from src.episodes import assign_episodes
from src.pipeline import run_detection
from src.scoring import ALERT, NO_ALERT, REVIEW, score_and_route

CFG = DetectorConfig()


def make_metrics(n=60, cid="T01", seed=0, noise=0.01):
    """Stable synthetic campaign with ~1% noise (below the 2% scale floor)."""
    rng = np.random.default_rng(seed)
    f = lambda base: base * (1 + rng.normal(0, noise, n))
    imp = np.round(f(100_000)).astype(int)
    clicks = np.round(imp * 0.02 * (1 + rng.normal(0, noise, n))).astype(int)
    conv = np.round(clicks * 0.05 * (1 + rng.normal(0, noise, n))).astype(int)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n), "campaign_id": cid,
        "spend": np.round(f(1000), 2), "impressions": imp, "clicks": clicks,
        "conversions": conv,
    })


def rec(records, metric, day, start="2026-01-01"):
    d = pd.Timestamp(start) + pd.Timedelta(days=day)
    return records[(records["metric"] == metric) & (records["date"] == d)].iloc[0]


@pytest.fixture(scope="module")
def bench():
    return run_detection(load_metrics())


# ---------- normal behaviour ----------
def test_normal_series_raises_no_flags():
    r, ep = run_detection(make_metrics())
    assert (r["route"] == NO_ALERT).all()
    assert ep.empty


# ---------- four incident types (controlled series) ----------
def test_spend_spike_alerts():
    m = make_metrics()
    m.loc[45, "spend"] *= 3
    r, _ = run_detection(m)
    x = rec(r, "spend", 45)
    assert x["route"] == ALERT and x["direction"] == "up" and x["severity"] == "high"
    assert "budget" in x["diagnostic_action"].lower()


def test_tracking_failure_alerts_on_conversions():
    m = make_metrics()
    m.loc[45:48, "conversions"] = 0
    r, _ = run_detection(m)
    x = rec(r, "conversions", 45)
    assert x["route"] == ALERT and x["direction"] == "down"
    assert "tag" in x["diagnostic_action"].lower()
    assert "does not by itself confirm the cause" in x["explanation"]


def test_ctr_collapse_flags_ctr():
    m = make_metrics()
    m.loc[45:47, "clicks"] = (m.loc[45:47, "clicks"] * 0.4).round().astype(int)
    m.loc[45:47, "conversions"] = (m.loc[45:47, "conversions"] * 0.4).round().astype(int)
    r, _ = run_detection(m)
    x = rec(r, "ctr", 45)
    assert x["route"] == ALERT and x["direction"] == "down"
    assert rec(r, "impressions", 45)["route"] == NO_ALERT  # volume unchanged


def test_gradual_degradation_eventually_flagged():
    m = make_metrics()
    factor = np.linspace(1.0, 0.5, 15)
    m.loc[45:59, "conversions"] = (m.loc[45:59, "conversions"] * factor).round().astype(int)
    r, _ = run_detection(m)
    late = r[(r["metric"].isin(["conversions", "cvr"]))
             & (r["date"] >= pd.Timestamp("2026-02-15")) & r["is_anomaly"]]
    assert len(late) > 0 and (late["direction"] == "down").all()


def test_dev_benchmark_incidents_are_flagged(bench):
    """Each dev-period incident has a flag on its affected metric (dev only)."""
    r, _ = bench
    expect = {"spend": "spend", "conversions": "conversions",
              "ctr": "ctr", "conversion_rate": "conversions"}
    for x in load_labels().query("period == 'dev'").itertuples():
        hit = r[(r["campaign_id"] == x.campaign_id) & r["date"].between(x.start_date, x.end_date)
                & (r["metric"] == expect[x.affected_metric]) & (r["route"] != NO_ALERT)]
        assert len(hit) > 0, x.incident_id


# ---------- leakage ----------
SCORE_COLS = ["observed", "baseline", "z", "evidence_score", "severity", "route", "explanation"]


def test_no_future_leakage():
    m = make_metrics()
    cut = pd.Timestamp("2026-02-05")
    m2 = m.copy()
    fut = m2["date"] > cut
    m2.loc[fut, ["spend", "impressions", "clicks", "conversions"]] *= 7
    a, _ = run_detection(m)
    b, _ = run_detection(m2)
    pa = a[a["date"] <= cut].reset_index(drop=True)[SCORE_COLS]
    pb = b[b["date"] <= cut].reset_index(drop=True)[SCORE_COLS]
    pd.testing.assert_frame_equal(pa, pb)
    assert not a[a["date"] > cut]["observed"].equals(b[b["date"] > cut]["observed"])


def test_detector_rejects_label_columns_and_never_loads_labels():
    m = make_metrics().assign(incident_type="x")
    with pytest.raises(ValueError):
        run_detection(m)
    src = inspect.getsource(pipeline)
    assert "load_labels" not in src and "load_benign" not in src


# ---------- edge cases ----------
def test_insufficient_history():
    r, _ = run_detection(make_metrics())
    early = r[r["date"] < pd.Timestamp("2026-01-01") + pd.Timedelta(days=CFG.min_history)]
    assert (early["status"] == "insufficient_history").all()
    assert early["z"].isna().all() and (early["route"] == NO_ALERT).all()
    assert "no baseline" in early.iloc[0]["explanation"]
    assert rec(r, "spend", CFG.min_history)["status"] == "scored"


def test_zero_variance_is_safe():
    m = make_metrics(noise=0.0)
    m[["spend", "impressions", "clicks", "conversions"]] = [1000.0, 100_000, 2000, 100]
    r, _ = run_detection(m)
    scored = r[r["status"] == "scored"]
    assert np.isfinite(scored["z"]).all() and (scored["route"] == NO_ALERT).all()
    m.loc[45, "spend"] = 1150.0  # 15% move on a flat series -> finite, flagged
    r, _ = run_detection(m)
    x = rec(r, "spend", 45)
    assert np.isfinite(x["z"]) and x["is_anomaly"]


def test_zero_denominators_are_safe():
    m = make_metrics()
    m.loc[30, ["impressions", "clicks", "conversions"]] = 0
    r, _ = run_detection(m)
    for metric in ["ctr", "cvr"]:
        x = rec(r, metric, 30)
        assert x["status"] == "missing_value" and x["route"] == NO_ALERT
    assert np.isfinite(r.loc[r["status"] == "scored", "z"]).all()


def test_missing_values_and_missing_days():
    m = make_metrics()
    m.loc[20, "spend"] = np.nan
    m = m.drop(index=25)  # whole day missing
    r, _ = run_detection(m)
    assert rec(r, "spend", 20)["status"] == "missing_value"
    assert rec(r, "clicks", 25)["status"] == "missing_value"  # reindexed gap
    later = rec(r, "spend", 40)
    assert later["status"] == "scored" and later["quality_score"] < 1.0


# ---------- evidence score and routing ----------
def test_evidence_bounds_and_composition(bench):
    r, _ = bench
    for c in ["magnitude_score", "history_score", "quality_score", "evidence_score"]:
        assert r[c].between(0, 1).all(), c
    prod = r["magnitude_score"] * r["history_score"] * r["quality_score"]
    assert np.allclose(r["evidence_score"], prod, atol=1e-4)
    assert (r.loc[~r["is_anomaly"], "route"] == NO_ALERT).all()


def _base(z, n_hist=28, completeness=1.0, clicks=2000):
    return pd.DataFrame({
        "campaign_id": "T01", "date": pd.Timestamp("2026-02-01"), "metric": "spend",
        "observed": 100.0, "baseline": 80.0, "z": [z], "n_history": [n_hist],
        "completeness": [completeness], "_impressions": [100_000.0],
        "_clicks": [float(clicks)], "_conversions": [100.0],
    })


def test_routing_rules():
    route = lambda **kw: score_and_route(_base(**kw), CFG).iloc[0]["route"]
    assert route(z=2.0) == NO_ALERT
    assert route(z=4.5) == REVIEW                          # borderline magnitude
    assert route(z=12.0) == ALERT                          # strong, full evidence
    assert route(z=12.0, n_hist=14) == ALERT               # history 0.5, magnitude 1 -> 0.5
    assert route(z=12.0, n_hist=14, completeness=0.8) == REVIEW  # weak history + quality
    assert route(z=-12.0, completeness=0.3) == REVIEW


# ---------- episodes ----------
def _flagged(rows):
    """rows: (day, metric, direction, severity, route)."""
    return pd.DataFrame([{
        "campaign_id": "T01", "date": pd.Timestamp("2026-02-01") + pd.Timedelta(days=d),
        "metric": m, "direction": dr, "severity": sev, "route": rt,
        "z": {"low": 4.5, "medium": 7.0, "high": 10.0}[sev] * (1 if dr == "up" else -1),
        "evidence_score": 0.6,
    } for d, m, dr, sev, rt in rows])


def test_episode_grouping_and_splitting():
    df = _flagged([
        (0, "spend", "up", "low", ALERT), (1, "spend", "up", "low", ALERT),
        (3, "spend", "up", "low", ALERT),     # 1 quiet day -> same episode
        (10, "spend", "up", "low", ALERT),    # long gap -> new episode
        (11, "spend", "down", "low", ALERT),  # direction flip -> new episode
        (1, "clicks", "up", "low", ALERT),    # other metric -> separate
    ])
    r, ep = assign_episodes(df, CFG)
    assert len(ep) == 4
    spend = r[r["metric"] == "spend"].sort_values("date")
    assert spend["episode_id"].iloc[:3].nunique() == 1
    assert spend["episode_id"].nunique() == 3
    assert list(spend["notification"].iloc[:3]) == ["new", "suppressed", "suppressed"]


def test_severity_escalation():
    df = _flagged([
        (0, "conversions", "down", "low", REVIEW),
        (1, "conversions", "down", "low", REVIEW),
        (2, "conversions", "down", "low", ALERT),    # route upgrade -> escalate
        (3, "conversions", "down", "high", ALERT),   # severity jump -> escalate
        (4, "conversions", "down", "medium", ALERT),  # not worse than notified
        (5, "conversions", "down", "high", ALERT),
    ])
    r, ep = assign_episodes(df, CFG)
    assert list(r.sort_values("date")["notification"]) == [
        "new", "suppressed", "escalation", "escalation", "suppressed", "suppressed"]
    assert len(ep) == 1 and ep.iloc[0]["notifications"] == 3
    assert ep.iloc[0]["peak_severity"] == "high" and ep.iloc[0]["top_route"] == ALERT


def test_benchmark_notifications_fewer_than_flags(bench):
    r, ep = bench
    flagged = (r["route"] != NO_ALERT).sum()
    sent = r["notification"].isin(["new", "escalation"]).sum()
    assert sent == ep["notifications"].sum()
    assert sent < flagged
