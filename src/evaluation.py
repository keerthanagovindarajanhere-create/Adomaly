"""Reproducible evaluation of the Adomaly detection pipeline.

Two datasets, evaluated differently and NEVER mixed:

- Synthetic benchmark (data/raw/campaign_metrics.csv + incident_labels.csv +
  benign_events.csv): has complete, documented ground truth. Every injected
  incident and every injected benign (non-incident) shift is listed. Because
  injection is exhaustive (src/data_generation.py plants every anomalous
  event the generator produces), a (campaign, metric, day) triple that is not
  inside a labelled incident window is a genuine negative, not an unlabeled
  unknown. Precision/recall/F1 and incident-level detection are therefore
  defined ONLY for this dataset, and ONLY over the chronological EVAL split
  (src/config.py DEV/EVAL): thresholds in src/detector_config.py were chosen
  on DEV, so reporting on DEV would be reporting on the tuning set.

- PS-04 organizer dataset (data/raw/daily_campaign_metrics.csv): has NO
  ground-truth labels at all. We cannot know which flagged records are real
  incidents, so precision, recall, F1 and "detection rate" are undefined and
  MUST NOT be computed here. Only alert-fatigue behaviour (how many raw
  flags collapse into how many notifications) is measured.

Matching rule (documented, not invented): an incident's `affected_metric`
is mapped to the single detector metric it is expected to show up on. This
reuses the exact mapping already used by
tests/test_detection.py::test_dev_benchmark_incidents_are_flagged, so the
evaluation is consistent with the project's own existing test suite:

    spend            -> spend
    conversions       -> conversions
    ctr               -> ctr
    conversion_rate    -> conversions   (gradual CVR decline suppresses the
                                         conversions count; the existing test
                                         already matches it this way)

A record-level evaluation unit is therefore (campaign_id, metric, date)
restricted to metric in {spend, conversions, ctr} -- the only metrics any
label targets. Flags on impressions, clicks, cvr or revenue are outside any
label's scope; they are reported separately as "unmapped-metric flags" and
are never counted as false positives, since there is no label claiming they
should (or shouldn't) fire.

A record is "detected"/flagged if its route is Alert or Human Review (both
represent a signal a human would see), not Alert alone -- Human Review is
still a successful catch at lower confidence, not a miss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.scoring import ALERT, NO_ALERT, REVIEW

METRIC_MAP = {
    "spend": "spend",
    "conversions": "conversions",
    "ctr": "ctr",
    "conversion_rate": "conversions",
}
MAPPED_METRICS = sorted(set(METRIC_MAP.values()))  # {"ctr", "conversions", "spend"}

FLAGGED_ROUTES = (ALERT, REVIEW)


# --------------------------------------------------------------- labeling --
def label_records(records: pd.DataFrame, labels: pd.DataFrame,
                   benign: pd.DataFrame) -> pd.DataFrame:
    """Attach ground-truth columns to detector output records (synthetic only).

    Adds:
      is_incident_positive, incident_id  -- inside a labelled incident window
      is_benign_shift, benign_id         -- inside a labelled benign window
    A record can be at most one of these; src/data_generation.py guarantees
    incident and benign campaigns never overlap
    (tests/test_data_generation.py::test_benign_events_do_not_overlap_incidents).
    """
    df = records.copy()
    df["is_incident_positive"] = False
    df["incident_id"] = pd.NA
    df["is_benign_shift"] = False
    df["benign_id"] = pd.NA

    for inc in labels.itertuples():
        metric = METRIC_MAP.get(inc.affected_metric)
        if metric is None:
            continue
        sel = ((df["campaign_id"] == inc.campaign_id) & (df["metric"] == metric)
               & df["date"].between(inc.start_date, inc.end_date))
        df.loc[sel, "is_incident_positive"] = True
        df.loc[sel, "incident_id"] = inc.incident_id

    for ben in benign.itertuples():
        metric = METRIC_MAP.get(ben.affected_metric, ben.affected_metric)
        sel = ((df["campaign_id"] == ben.campaign_id) & (df["metric"] == metric)
               & (df["date"] >= ben.start_date) & ~df["is_incident_positive"])
        df.loc[sel, "is_benign_shift"] = True
        df.loc[sel, "benign_id"] = ben.event_id

    return df


# ------------------------------------------------------- record-level P/R/F1
@dataclass
class RecordLevelResult:
    n_evaluated: int
    n_positive: int
    n_negative: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    n_unmapped_metric_flags: int  # informational, not counted above


def record_level_metrics(labelled: pd.DataFrame) -> RecordLevelResult:
    """Precision/recall/F1 over (campaign, metric, date) restricted to
    MAPPED_METRICS -- the only metrics any label targets. `labelled` should
    already be filtered to the split being evaluated (e.g. eval-only)."""
    universe = labelled[labelled["metric"].isin(MAPPED_METRICS)]
    flagged = universe["route"].isin(FLAGGED_ROUTES)
    positive = universe["is_incident_positive"]

    tp = int((flagged & positive).sum())
    fn = int((~flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    tn = int((~flagged & ~positive).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) > 0 else float("nan"))

    unmapped_flags = int((labelled["route"].isin(FLAGGED_ROUTES)
                          & ~labelled["metric"].isin(MAPPED_METRICS)).sum())

    return RecordLevelResult(
        n_evaluated=len(universe), n_positive=int(positive.sum()),
        n_negative=int((~positive).sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1,
        n_unmapped_metric_flags=unmapped_flags,
    )


# ------------------------------------------------------ incident-level view
def incident_level_results(records: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled incident: whether any flagged record fell inside
    its window on the mapped metric, and detection delay in days (first
    flagged day minus incident start_date). Incidents never detected have
    delay_days = NaN."""
    rows = []
    for inc in labels.itertuples():
        metric = METRIC_MAP.get(inc.affected_metric)
        hits = records[(records["campaign_id"] == inc.campaign_id)
                       & (records["metric"] == metric)
                       & records["date"].between(inc.start_date, inc.end_date)
                       & records["route"].isin(FLAGGED_ROUTES)]
        detected = len(hits) > 0
        delay = (hits["date"].min() - inc.start_date).days if detected else float("nan")
        rows.append({
            "incident_id": inc.incident_id, "campaign_id": inc.campaign_id,
            "incident_type": inc.incident_type, "mapped_metric": metric,
            "severity": inc.severity, "period": inc.period,
            "start_date": inc.start_date, "end_date": inc.end_date,
            "detected": detected, "first_flag_date": hits["date"].min() if detected else pd.NaT,
            "delay_days": delay,
            "best_route": (ALERT if (hits["route"] == ALERT).any() else REVIEW) if detected else None,
        })
    return pd.DataFrame(rows)


def false_alerts_on_benign(labelled: pd.DataFrame) -> dict:
    """Flags raised on labelled benign (known non-incident) shifts -- the
    exact false-positive scenario PS-04 and the synthetic benchmark are both
    designed to avoid rewarding: don't keep alerting on a planned, sustained
    change."""
    shift = labelled[labelled["is_benign_shift"]]
    flagged = shift[shift["route"].isin(FLAGGED_ROUTES)]
    return {
        "benign_shift_records": int(len(shift)),
        "benign_shift_flagged_records": int(len(flagged)),
        "benign_shift_alert_records": int((flagged["route"] == ALERT).sum()),
        "benign_shift_flag_rate": (len(flagged) / len(shift)) if len(shift) else float("nan"),
        "benign_events_with_any_flag": sorted(flagged["benign_id"].dropna().unique().tolist()),
    }


# ----------------------------------------------------------- alert fatigue
def alert_fatigue_summary(records: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """Dataset-agnostic: works for the synthetic benchmark and for PS-04
    (no labels needed -- this only measures volume reduction, not correctness)."""
    flagged = records[records["route"].isin(FLAGGED_ROUTES)]
    notes = flagged["notification"].value_counts()
    n_new = int(notes.get("new", 0))
    n_esc = int(notes.get("escalation", 0))
    n_sup = int(notes.get("suppressed", 0))
    sent = n_new + n_esc
    raw = len(flagged)
    return {
        "raw_flagged_records": raw,
        "alerts": int((flagged["route"] == ALERT).sum()),
        "human_review_records": int((flagged["route"] == REVIEW).sum()),
        "episodes": int(len(episodes)),
        "notifications_sent": sent,
        "new_episode_notifications": n_new,
        "escalation_notifications": n_esc,
        "suppressed_repeats": n_sup,
        "notification_reduction_pct": ((raw - sent) / raw * 100.0) if raw else float("nan"),
    }


# ===================================================================
# Round-2 additions: notification-level coverage and system comparison.
# Everything below is additive; the functions above are unchanged.
# ===================================================================
NOTIFYING = ("new", "escalation")


def notify_every_flag(records: pd.DataFrame) -> pd.DataFrame:
    """The "adaptive detector, every flag notified" comparison system.

    Same detector, same flagged records as full Adomaly, but with no
    evidence routing (every flag is an Alert) and no episode grouping
    (every flag is its own "new" notification). Unflagged records are
    untouched. Used to isolate what routing + episodes add.
    """
    df = records.copy()
    flagged = df["route"].isin(FLAGGED_ROUTES)
    df.loc[flagged, "route"] = ALERT
    df["notification"] = np.where(flagged, "new", "")
    if "episode_id" in df.columns:
        df["episode_id"] = pd.NA
    return df


def notification_coverage(records: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled incident: did a human actually get NOTIFIED?

    Credit rule: an incident is `notified` only if at least one record on its
    mapped metric, same campaign, with date inside [start_date, end_date], has
    notification "new" or "escalation". Flags that were suppressed do not count.

    status:
      notified               -- >= 1 new/escalation notification in window
      detected_not_notified  -- flagged in window, but every flag suppressed
      missed                 -- no flag in window at all

    Also reports notifications_in_window (fragmentation: >1 means the same
    incident notified more than once), suppressed_in_window,
    episode_started_before_window (an in-window flag belongs to an episode
    whose first flag predates the window -- the case where suppression can
    hide an incident), and review_only (detected, but every in-window flag
    was Human Review -- an Alert-only policy would not have surfaced it).
    """
    has_ep = "episode_id" in records.columns
    flagged_all = records[records["route"].isin(FLAGGED_ROUTES)]
    rows = []
    for inc in labels.itertuples():
        metric = METRIC_MAP.get(inc.affected_metric)
        same = flagged_all[(flagged_all["campaign_id"] == inc.campaign_id)
                           & (flagged_all["metric"] == metric)]
        win = same[same["date"].between(inc.start_date, inc.end_date)]
        notes = win[win["notification"].isin(NOTIFYING)]
        n_flags, n_notes = len(win), len(notes)
        status = "notified" if n_notes else ("detected_not_notified" if n_flags else "missed")

        started_before = False
        if has_ep and n_flags:
            eps = win["episode_id"].dropna().unique()
            if len(eps):
                first = same[same["episode_id"].isin(eps)].groupby("episode_id")["date"].min()
                started_before = bool((first < inc.start_date).any())

        first_flag = win["date"].min() if n_flags else pd.NaT
        first_note = notes["date"].min() if n_notes else pd.NaT
        rows.append({
            "incident_id": inc.incident_id, "campaign_id": inc.campaign_id,
            "incident_type": inc.incident_type, "mapped_metric": metric,
            "severity": inc.severity, "period": inc.period,
            "start_date": inc.start_date, "end_date": inc.end_date,
            "status": status, "detected": n_flags > 0, "notified": n_notes > 0,
            "flags_in_window": n_flags, "notifications_in_window": n_notes,
            "suppressed_in_window": int((win["notification"] == "suppressed").sum()),
            "episode_started_before_window": started_before,
            "review_only": bool(n_flags and (win["route"] == REVIEW).all()),
            "first_flag_date": first_flag, "first_notification_date": first_note,
            "detection_delay_days": (first_flag - inc.start_date).days if n_flags else np.nan,
            "notification_delay_days": (first_note - inc.start_date).days if n_notes else np.nan,
        })
    return pd.DataFrame(rows)


def route_precision(labelled: pd.DataFrame) -> dict:
    """Precision of flags by route, over the labelled universe (MAPPED_METRICS).
    `labelled` should already be filtered to the split being evaluated.
    Shows whether evidence-based routing separates stronger from weaker cases."""
    u = labelled[labelled["metric"].isin(MAPPED_METRICS)]
    out = {}
    for route in FLAGGED_ROUTES:
        f = u[u["route"] == route]
        tp = int(f["is_incident_positive"].sum())
        out[route] = {"flags": int(len(f)), "tp": tp, "fp": int(len(f) - tp),
                      "precision": tp / len(f) if len(f) else float("nan")}
    return out


def notification_burden(records: pd.DataFrame) -> dict:
    """Dataset-agnostic volume over ALL scored metrics (works for PS-04).
    No labels involved, so no accuracy claim."""
    flagged = records[records["route"].isin(FLAGGED_ROUTES)]
    raw = int(len(flagged))
    sent = int(flagged["notification"].isin(NOTIFYING).sum())
    return {"raw_flagged_records": raw, "notifications_sent": sent,
            "suppressed_repeats": int((flagged["notification"] == "suppressed").sum()),
            "notification_reduction_pct": ((raw - sent) / raw * 100.0) if raw else float("nan")}


def system_metrics(name: str, records: pd.DataFrame, labels: pd.DataFrame,
                   benign: pd.DataFrame, period: str = "eval") -> dict:
    """All comparison measures for one system on one split (synthetic only).

    - precision / recall / F1 / FPR: record level over MAPPED_METRICS
      (record_level_metrics), FPR = FP / (FP + TN).
    - incident detection and notification coverage: notification_coverage,
      restricted to incidents of this split.
    - notification burden: ALL detector metrics in this split (records whose
      `period` == period), so it measures what an operator would receive.
    - notification_precision: share of notifications on MAPPED_METRICS that
      fall inside a labelled incident window.
    """
    recs = records[records["period"] == period]
    labelled = label_records(recs, labels, benign)
    rl = record_level_metrics(labelled)
    # Coverage uses ALL records so an episode that began before this split is
    # still recognised; only this split's incidents are scored.
    cov = notification_coverage(records, labels[labels["period"] == period])
    burden = notification_burden(recs)
    fa = false_alerts_on_benign(labelled)

    mapped_notes = labelled[labelled["metric"].isin(MAPPED_METRICS)
                            & labelled["route"].isin(FLAGGED_ROUTES)
                            & labelled["notification"].isin(NOTIFYING)]
    n_inc = len(cov)
    det = cov["detected"].sum() if n_inc else 0
    notified = cov["notified"].sum() if n_inc else 0
    nan = float("nan")
    return {
        "system": name, "period": period,
        "precision": rl.precision, "recall": rl.recall, "f1": rl.f1,
        "false_positive_rate": rl.fp / (rl.fp + rl.tn) if (rl.fp + rl.tn) else nan,
        "tp": rl.tp, "fp": rl.fp, "fn": rl.fn, "tn": rl.tn,
        "n_incidents": int(n_inc),
        "incidents_detected": int(det),
        "incidents_notified": int(notified),
        "incidents_detected_not_notified": int((cov["status"] == "detected_not_notified").sum()) if n_inc else 0,
        "incidents_missed": int((cov["status"] == "missed").sum()) if n_inc else 0,
        "incidents_reaching_human_via_review_only": int(cov["review_only"].sum()) if n_inc else 0,
        "incidents_with_multiple_notifications": int((cov["notifications_in_window"] > 1).sum()) if n_inc else 0,
        "notifications_in_incident_windows": int(cov["notifications_in_window"].sum()) if n_inc else 0,
        "incident_detection_rate": det / n_inc if n_inc else nan,
        "notification_coverage_rate": notified / n_inc if n_inc else nan,
        "median_detection_delay_days": float(cov["detection_delay_days"].median()) if det else nan,
        "median_notification_delay_days": float(cov["notification_delay_days"].median()) if notified else nan,
        **burden,
        # Counts behind notification_precision. Suppression removes repeat
        # in-incident days as well as repeat false days, so the ratio alone
        # can fall even when false notifications fall; report both counts.
        "notifications_on_labelled_metrics": int(len(mapped_notes)),
        "true_incident_notifications": int(mapped_notes["is_incident_positive"].sum()),
        "false_notifications": int((~mapped_notes["is_incident_positive"].astype(bool)).sum()),
        "notification_precision": (mapped_notes["is_incident_positive"].mean()
                                   if len(mapped_notes) else nan),
        "benign_shift_flagged_records": fa["benign_shift_flagged_records"],
    }
