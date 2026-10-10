import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_generation import generate_dataset, save_dataset
from src.data_io import assign_period, load_labels, load_metrics


@pytest.fixture(scope="module")
def data():
    return generate_dataset()


# ---------- schema ----------
def test_metrics_schema(data):
    m, _, _ = data
    assert list(m.columns) == C.METRIC_COLUMNS
    assert len(m) == C.N_CAMPAIGNS * C.N_DAYS
    assert m["campaign_id"].nunique() == C.N_CAMPAIGNS
    assert m["date"].min() == C.START_DATE and m["date"].max() == C.END_DATE
    assert not m.duplicated(["campaign_id", "date"]).any()
    assert m.notna().all().all()


def test_label_schema(data):
    _, lab, _ = data
    assert list(lab.columns) == C.LABEL_COLUMNS
    assert lab["incident_id"].is_unique
    assert set(lab["incident_type"]) == set(C.INCIDENT_TYPES)
    assert set(lab["severity"]) <= {"low", "medium", "high"}


def test_metrics_contain_no_label_information(data):
    m, lab, _ = data
    assert not set(m.columns) & (set(C.LABEL_COLUMNS) - {"campaign_id"})


# ---------- reproducibility ----------
def test_reproducible_with_same_seed():
    a, b = generate_dataset(), generate_dataset()
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_different_seed_changes_metrics_not_labels():
    m1, l1, _ = generate_dataset(seed=1)
    m2, l2, _ = generate_dataset(seed=2)
    assert not m1["spend"].equals(m2["spend"])
    pd.testing.assert_frame_equal(l1, l2)


def test_saved_files_roundtrip(tmp_path, monkeypatch, data):
    for name in ["RAW_DIR", "METRICS_FILE", "LABELS_FILE", "BENIGN_FILE"]:
        monkeypatch.setattr(C, name, tmp_path / getattr(C, name).name)
    save_dataset(*data)
    m = load_metrics(C.METRICS_FILE)
    expected = data[0].sort_values(["campaign_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(m, expected, check_dtype=False)
    assert len(load_labels(C.LABELS_FILE)) == len(data[1])


def test_loader_rejects_leaked_label_columns(tmp_path, data):
    p = tmp_path / "bad.csv"
    data[0].assign(incident_type="x").to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_metrics(p)


# ---------- consistency ----------
def test_metric_relationships(data):
    m, _, _ = data
    assert (m["spend"] > 0).all()
    assert (m[["impressions", "clicks", "conversions"]] >= 0).all().all()
    assert (m["clicks"] <= m["impressions"]).all()
    assert (m["conversions"] <= m["clicks"]).all()
    ctr = m["clicks"] / m["impressions"]
    assert ctr.between(0.001, 0.08).all()
    cpm = m["spend"] / m["impressions"] * 1000
    assert cpm.between(2, 25).all()


def test_weekday_pattern_present(data):
    m, _, _ = data
    m = m.assign(weekend=m["date"].dt.dayofweek >= 5)
    ratio = (m.groupby(["campaign_id", "weekend"])["spend"].median().unstack())
    ratio = ratio[True] / ratio[False]
    assert (ratio.sub(1).abs() > 0.05).sum() >= 8  # most campaigns show weekend effect


# ---------- incident labels ----------
def test_incidents_valid_and_split_respected(data):
    m, lab, _ = data
    s, e = pd.to_datetime(lab["start_date"]), pd.to_datetime(lab["end_date"])
    assert (s <= e).all()
    assert (s >= C.START_DATE + pd.Timedelta(days=C.WARMUP_DAYS)).all()
    assert (e <= C.END_DATE).all()
    assert set(lab["campaign_id"]) <= set(m["campaign_id"])
    # no incident crosses the dev/eval boundary
    assert (assign_period(s) == assign_period(e)).all()
    assert (assign_period(s) == lab["period"]).all()
    # every incident type appears in both periods
    for period in ["dev", "eval"]:
        assert set(lab.loc[lab["period"] == period, "incident_type"]) == set(C.INCIDENT_TYPES)


def test_no_overlapping_incidents_per_campaign(data):
    _, lab, _ = data
    for _, g in lab.assign(s=pd.to_datetime(lab["start_date"]),
                           e=pd.to_datetime(lab["end_date"])).groupby("campaign_id"):
        g = g.sort_values("s")
        assert (g["s"].iloc[1:].to_numpy() > g["e"].iloc[:-1].to_numpy()).all()


def _window(m, cid, start, end):
    g = m[m["campaign_id"] == cid].set_index("date")
    pre = g.loc[pd.Timestamp(start) - pd.Timedelta(days=28): pd.Timestamp(start) - pd.Timedelta(days=1)]
    inc = g.loc[start:end]
    return pre, inc


def test_incidents_visible_in_metrics(data):
    """Each injected incident moves its affected metric in the intended direction."""
    m, lab, _ = data
    for r in lab.itertuples():
        pre, inc = _window(m, r.campaign_id, r.start_date, r.end_date)
        if r.incident_type == "spend_spike":
            assert inc["spend"].mean() > 1.3 * pre["spend"].median(), r.incident_id
        elif r.incident_type == "tracking_failure":
            pre_cvr = pre["conversions"].sum() / pre["clicks"].sum()
            inc_cvr = inc["conversions"].sum() / inc["clicks"].sum()
            assert inc_cvr < 0.7 * pre_cvr, r.incident_id
        elif r.incident_type == "ctr_collapse":
            pre_ctr = pre["clicks"].sum() / pre["impressions"].sum()
            inc_ctr = inc["clicks"].sum() / inc["impressions"].sum()
            assert inc_ctr < 0.85 * pre_ctr, r.incident_id
        elif r.incident_type == "gradual_degradation":
            tail = inc.iloc[-7:]
            pre_cvr = pre["conversions"].sum() / pre["clicks"].sum()
            tail_cvr = tail["conversions"].sum() / tail["clicks"].sum()
            assert tail_cvr < 0.85 * pre_cvr, r.incident_id


def test_benign_events_do_not_overlap_incidents(data):
    _, lab, ben = data
    assert not set(ben["campaign_id"]) & set(lab["campaign_id"])
