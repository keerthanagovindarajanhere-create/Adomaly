"""Edge cases for src/episodes.py not already covered by
tests/test_detection.py (test_episode_grouping_and_splitting,
test_severity_escalation). Those existing tests are left unchanged."""
import pandas as pd

from src.detector_config import DetectorConfig
from src.episodes import assign_episodes
from src.scoring import ALERT, NO_ALERT, REVIEW

CFG = DetectorConfig()
Z = {"low": 4.5, "medium": 7.0, "high": 10.0}


def _rows(rows, campaign="T01"):
    """rows: (day, metric, direction, severity, route)."""
    return pd.DataFrame([{
        "campaign_id": campaign,
        "date": pd.Timestamp("2026-02-01") + pd.Timedelta(days=d),
        "metric": m, "direction": dr, "severity": sev, "route": rt,
        "z": Z.get(sev, 0.0) * (1 if dr == "up" else -1),
        "evidence_score": 0.6 if rt == ALERT else 0.4,
    } for d, m, dr, sev, rt in rows])


def test_gap_exactly_one_more_than_allowed_starts_new_episode():
    # max_gap_days = 1: days 0 and 2 (1 quiet day) join; day 5 (2 quiet days) splits.
    assert CFG.max_gap_days == 1
    df = _rows([(0, "spend", "up", "low", ALERT), (2, "spend", "up", "low", ALERT),
                (5, "spend", "up", "low", ALERT)])
    r, ep = assign_episodes(df, CFG)
    r = r.sort_values("date")
    assert r["episode_id"].iloc[0] == r["episode_id"].iloc[1]
    assert r["episode_id"].iloc[2] != r["episode_id"].iloc[1]
    assert list(r["notification"]) == ["new", "suppressed", "new"]
    assert len(ep) == 2


def test_zero_gap_config_splits_on_any_quiet_day():
    cfg = DetectorConfig(max_gap_days=0)
    df = _rows([(0, "ctr", "down", "low", ALERT), (1, "ctr", "down", "low", ALERT),
                (3, "ctr", "down", "low", ALERT)])
    _, ep = assign_episodes(df, cfg)
    assert len(ep) == 2


def test_alert_to_review_downgrade_is_suppressed_not_escalated():
    df = _rows([(0, "conversions", "down", "medium", ALERT),
                (1, "conversions", "down", "medium", REVIEW),
                (2, "conversions", "down", "low", REVIEW)])
    r, ep = assign_episodes(df, CFG)
    assert list(r.sort_values("date")["notification"]) == ["new", "suppressed", "suppressed"]
    assert ep.iloc[0]["notifications"] == 1 and ep.iloc[0]["top_route"] == ALERT


def test_review_only_episode_reports_review_as_top_route():
    df = _rows([(0, "spend", "up", "low", REVIEW), (1, "spend", "up", "low", REVIEW)])
    _, ep = assign_episodes(df, CFG)
    assert len(ep) == 1 and ep.iloc[0]["top_route"] == REVIEW


def test_same_metric_different_campaigns_are_separate_episodes():
    df = pd.concat([_rows([(0, "spend", "up", "low", ALERT)], "T01"),
                    _rows([(0, "spend", "up", "low", ALERT)], "T02")], ignore_index=True)
    r, ep = assign_episodes(df, CFG)
    assert len(ep) == 2 and (r["notification"] == "new").all()


def test_unflagged_records_get_no_episode_or_notification():
    df = _rows([(0, "spend", "up", "low", ALERT), (1, "spend", "flat", "none", NO_ALERT)])
    r, ep = assign_episodes(df, CFG)
    quiet = r[r["route"] == NO_ALERT].iloc[0]
    assert pd.isna(quiet["episode_id"]) and quiet["notification"] == ""
    assert len(ep) == 1


def test_no_flags_gives_empty_episode_table_with_schema():
    df = _rows([(0, "spend", "flat", "none", NO_ALERT)])
    r, ep = assign_episodes(df, CFG)
    assert ep.empty and "notifications" in ep.columns and "suppressed" in ep.columns
    assert (r["notification"] == "").all()


def test_summary_counts_add_up_to_flagged_days():
    df = _rows([(0, "spend", "up", "low", REVIEW), (1, "spend", "up", "low", ALERT),
                (2, "spend", "up", "high", ALERT), (3, "spend", "up", "medium", ALERT),
                (10, "spend", "down", "low", ALERT)])
    r, ep = assign_episodes(df, CFG)
    assert (ep["notifications"] + ep["suppressed"] == ep["flagged_days"]).all()
    sent = r["notification"].isin(["new", "escalation"]).sum()
    assert sent == ep["notifications"].sum()
    assert (r["notification"] == "suppressed").sum() == ep["suppressed"].sum()
