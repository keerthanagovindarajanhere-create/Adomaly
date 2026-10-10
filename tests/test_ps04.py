"""Tests for the PS-04 supplied dataset integration.

Kept in its own file, separate from tests/test_detection.py and
tests/test_data_generation.py, which continue to cover the synthetic
benchmark exactly as before.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import pipeline
from src.data_io import load_metrics, load_ps04_metrics
from src.detector_config import DetectorConfig, PS04_BASE_METRICS, RATIO_METRICS
from src.pipeline import run_detection, run_ps04_detection, run_ps04_pipeline
from src.scoring import ALERT, NO_ALERT, REVIEW

SMALL_CAMPAIGNS = ["C03", "C11", "C14", "C15", "C19"]  # low-volume / noisy by inspection
B2B_CAMPAIGNS = ["C13", "C14", "C15", "C16"]


@pytest.fixture(scope="module")
def ps04():
    m, meta = load_ps04_metrics()
    return m, meta


@pytest.fixture(scope="module")
def ps04_bench(ps04):
    m, meta = ps04
    return run_ps04_detection(m, meta)


# ---------- 1. schema loading ----------
def test_ps04_schema_loaded_correctly(ps04):
    m, meta = ps04
    assert list(m.columns) == C.PS04_METRIC_COLUMNS
    assert len(m) == 20 * 180
    assert m["campaign_id"].nunique() == 20
    assert not m.duplicated(["campaign_id", "date"]).any()
    assert m.notna().all().all()
    assert (m["clicks"] <= m["impressions"]).all()
    assert (m["conversions"] <= m["clicks"]).all()

    assert list(meta.columns) == C.PS04_METADATA_COLUMNS
    assert meta["campaign_id"].is_unique and len(meta) == 20
    assert set(meta["vertical"]) == {"ecommerce", "b2b"}


def test_ps04_loader_rejects_wrong_schema(tmp_path, ps04):
    m, _ = ps04
    bad = m.drop(columns=["revenue"])  # missing a required column
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_ps04_metrics(p)


# ---------- 2. B2B zero revenue stays non-anomalous ----------
def test_b2b_zero_revenue_zero_baseline_and_mad_is_safe():
    """Direct unit check: an all-zero revenue series gives baseline=0, MAD=0,
    z=0 (finite, via the scale floor) and is never flagged."""
    cfg = DetectorConfig(metrics=PS04_BASE_METRICS + tuple(RATIO_METRICS))
    n = 60
    m = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=n), "campaign_id": "B01",
        "spend": 500.0, "impressions": 50_000, "clicks": 300, "conversions": 15,
        "revenue": 0.0,
    })
    r, _ = run_detection(m, cfg=cfg, allowed_columns=C.PS04_METRIC_COLUMNS)
    rev = r[(r["metric"] == "revenue") & (r["status"] == "scored")]
    assert len(rev) > 0
    assert (rev["baseline"] == 0).all()
    assert np.isfinite(rev["z"]).all() and (rev["z"] == 0).all()
    assert (rev["route"] == NO_ALERT).all()
    assert not rev["is_anomaly"].any()


def test_b2b_revenue_never_anomalous_on_real_data(ps04_bench):
    r, _ = ps04_bench
    b2b_revenue = r[(r["vertical"] == "b2b") & (r["metric"] == "revenue")]
    assert len(b2b_revenue) > 0
    scored = b2b_revenue[b2b_revenue["status"] == "scored"]
    assert len(scored) > 0
    assert (scored["baseline"] == 0).all()
    assert np.isfinite(scored["z"]).all() and (scored["z"] == 0).all()
    assert not b2b_revenue["is_anomaly"].any()
    assert (b2b_revenue["route"] == NO_ALERT).all()


# ---------- 3. normal weekend variation does not cause excessive alerts ----------
def test_weekend_variation_does_not_cause_excessive_alerts(ps04_bench):
    r, _ = ps04_bench
    b2b = r[(r["vertical"] == "b2b") & (r["status"] == "scored")].copy()
    assert len(b2b) > 0
    b2b["weekend"] = b2b["date"].dt.dayofweek >= 5
    rate = b2b.groupby("weekend")["is_anomaly"].mean()
    # B2B campaigns halve in volume at weekends by design (see README); if the
    # weekday adjustment were not working, every weekend would alert (rate -> 1).
    assert rate.get(True, 0.0) < 0.05
    assert rate.get(False, 0.0) < 0.05


# ---------- 4. small/noisy campaigns stay finite with sensible routing ----------
def test_small_campaigns_scores_finite_and_routing_sensible(ps04_bench):
    r, _ = ps04_bench
    small = r[r["campaign_id"].isin(SMALL_CAMPAIGNS)]
    scored = small[small["status"] == "scored"]
    assert len(scored) > 0
    assert np.isfinite(scored["z"]).all()
    assert np.isfinite(scored["evidence_score"]).all()
    assert scored["evidence_score"].between(0, 1).all()
    assert set(small["route"].unique()) <= {NO_ALERT, REVIEW, ALERT}
    assert (scored.loc[~scored["is_anomaly"], "route"] == NO_ALERT).all()


# ---------- 5. ground-truth labels never required or used ----------
def test_ps04_path_never_loads_or_requires_labels():
    src = inspect.getsource(pipeline)
    assert "load_labels" not in src and "load_benign" not in src
    # The organizers supplied no labels for this dataset at all.
    assert not (C.RAW_DIR / "daily_campaign_metrics_labels.csv").exists()


def test_ps04_metadata_cannot_be_passed_to_detector_directly(ps04):
    """Metadata columns must be split out via load_ps04_metrics before the
    detector sees anything; passing the raw file's columns straight through
    is rejected, same as the synthetic benchmark's anti-leakage guard."""
    raw = pd.read_csv(C.PS04_FILE, parse_dates=["date"])
    with pytest.raises(ValueError):
        run_detection(raw, allowed_columns=C.PS04_METRIC_COLUMNS)


def test_synthetic_guard_unaffected_by_ps04_columns():
    """The default (synthetic) column guard is unchanged: PS-04-shaped data
    is rejected by the default allowed_columns, exactly as label columns are."""
    raw = pd.read_csv(C.PS04_FILE, parse_dates=["date"])
    with pytest.raises(ValueError):
        run_detection(raw)  # no allowed_columns override -> synthetic default


# ---------- 6. synthetic pipeline behaves identically after the changes ----------
def test_synthetic_pipeline_unchanged_and_deterministic():
    m = load_metrics()
    r1, ep1 = run_detection(m)
    r2, ep2 = run_detection(m)
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(ep1, ep2)
    assert list(r1.columns) == pipeline.RECORD_COLUMNS


def test_ps04_pipeline_entrypoint_matches_manual_call(ps04, ps04_bench):
    m, meta = ps04
    r_direct, ep_direct = run_ps04_pipeline()
    r_manual, ep_manual = ps04_bench
    pd.testing.assert_frame_equal(r_direct, r_manual)
    pd.testing.assert_frame_equal(ep_direct, ep_manual)
