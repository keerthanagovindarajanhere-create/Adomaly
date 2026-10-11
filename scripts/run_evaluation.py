"""Reproducible evaluation report for Adomaly.

Run:  python -m scripts.run_evaluation

Produces, under reports/evaluation/:
  report.md                      - human-readable report with definitions,
                                    assumptions and limitations
  synthetic_metrics.json         - record-level P/R/F1 + fatigue (eval split)
  synthetic_incident_results.csv - per-incident detection + delay
  synthetic_false_alerts.json    - flags on labelled benign shifts
  ps04_metrics.json              - alert-fatigue only (no labels exist)
  alert_fatigue_comparison.csv   - synthetic vs PS-04, side by side

Round-2 controlled comparison (added; files above are unchanged):
  system_comparison.csv              - fixed vs adaptive vs full Adomaly,
                                        synthetic EVAL split, one row each
  system_comparison.json             - same + threshold selection + route precision
  fixed_threshold_grid.csv           - DEV-only threshold search for the baseline
  incident_notification_coverage.csv - per incident x system: notified / detected
                                        but not notified / missed
  route_precision.csv                - flag precision by route, per system (EVAL)
  synthetic_incident_replay.csv      - day-by-day records around each EVAL incident
                                        for the dashboard replay
  ps04_notification_burden.csv       - PS-04 volume only, per system (no labels)

Uses the existing detection pipeline and existing default thresholds
(src/detector_config.py DetectorConfig()) exactly as shipped; this script
does not tune or alter them. The fixed baseline's single threshold is chosen
on the synthetic DEV period only (src/baselines.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.baselines import fixed_threshold_records, select_fixed_threshold
from src.data_io import load_benign, load_labels, load_metrics, load_ps04_metrics
from src.evaluation import (
    METRIC_MAP,
    alert_fatigue_summary,
    false_alerts_on_benign,
    incident_level_results,
    label_records,
    notification_burden,
    notification_coverage,
    notify_every_flag,
    record_level_metrics,
    route_precision,
    system_metrics,
)
from src.pipeline import run_detection, run_ps04_detection

REPORT_DIR = C.ROOT / "reports" / "evaluation"

# Order is the display order everywhere (CSV rows, report, dashboard).
SYSTEMS = ("fixed_threshold", "adaptive_detector", "full_adomaly")
SYSTEM_LABELS = {
    "fixed_threshold": "Fixed threshold (day-over-day)",
    "adaptive_detector": "Adaptive detector (every flag notified)",
    "full_adomaly": "Full Adomaly (routing + episodes)",
}
REPLAY_PAD_DAYS = 7
REPLAY_COLS = ["date", "campaign_id", "metric", "observed", "baseline", "z", "severity",
               "evidence_score", "route", "notification", "episode_id", "explanation"]


def _fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def _fmt(x) -> str:
    return "n/a" if (isinstance(x, float) and pd.isna(x)) else str(x)


def run_synthetic_evaluation():
    metrics = load_metrics()
    labels = load_labels()
    benign = load_benign()
    records, episodes = run_detection(metrics)

    labelled = label_records(records, labels, benign)
    eval_labelled = labelled[labelled["period"] == "eval"]
    eval_records = records[records["period"] == "eval"]
    eval_episodes = episodes  # episodes aren't period-tagged; see note in report

    rl = record_level_metrics(eval_labelled)
    inc_results = incident_level_results(records, labels)  # all periods, for reference
    inc_results_eval = inc_results[inc_results["period"] == "eval"]
    false_alerts = false_alerts_on_benign(labelled[labelled["period"] == "eval"])
    fatigue = alert_fatigue_summary(records, episodes)
    fatigue_eval = alert_fatigue_summary(eval_records,
                                          episodes[episodes["end_date"] >= C.EVAL_START])

    detection_rate_eval = (inc_results_eval["detected"].mean()
                           if len(inc_results_eval) else float("nan"))
    delays = inc_results_eval.loc[inc_results_eval["detected"], "delay_days"]

    return {
        "record_level": rl,
        "incident_results_all": inc_results,
        "incident_results_eval": inc_results_eval,
        "detection_rate_eval": detection_rate_eval,
        "mean_delay_days_eval": float(delays.mean()) if len(delays) else float("nan"),
        "median_delay_days_eval": float(delays.median()) if len(delays) else float("nan"),
        "false_alerts_eval": false_alerts,
        "fatigue_full": fatigue,
        "fatigue_eval": fatigue_eval,
        "n_dev_incidents": int((labels["period"] == "dev").sum()),
        "n_eval_incidents": int((labels["period"] == "eval").sum()),
        # In-memory only (never serialized): reused by run_comparison so the
        # detector is not run twice. write_outputs ignores these keys.
        "_metrics": metrics, "_labels": labels, "_benign": benign,
        "_records": records,
    }


def run_ps04_evaluation(keep_records: bool = False):
    """Summary dict is unchanged (it is written verbatim to ps04_metrics.json).
    With keep_records=True also returns (summary, metrics, records)."""
    metrics, metadata = load_ps04_metrics()
    records, episodes = run_ps04_detection(metrics, metadata)
    fatigue = alert_fatigue_summary(records, episodes)
    summary = {"fatigue": fatigue, "n_campaigns": int(metrics["campaign_id"].nunique()),
               "n_records": int(len(records))}
    return (summary, metrics, records) if keep_records else summary


# ===================================================== Round-2 comparison ==
def build_replay(full: pd.DataFrame, fixed: pd.DataFrame, labels: pd.DataFrame,
                 pad: int = REPLAY_PAD_DAYS) -> pd.DataFrame:
    """Day-by-day view of each EVAL incident's mapped metric, padded by `pad`
    days each side. Detector columns come from full Adomaly's records; the
    fixed baseline's route/notification on the same day are joined alongside.
    Labels are used only to choose the window; nothing here feeds a detector."""
    keep = [c for c in REPLAY_COLS if c in full.columns]
    frames = []
    for inc in labels[labels["period"] == "eval"].itertuples():
        metric = METRIC_MAP.get(inc.affected_metric)
        if metric is None:
            continue
        lo = inc.start_date - pd.Timedelta(days=pad)
        hi = inc.end_date + pd.Timedelta(days=pad)
        f = full[(full["campaign_id"] == inc.campaign_id) & (full["metric"] == metric)
                 & full["date"].between(lo, hi)][keep]
        b = fixed[(fixed["campaign_id"] == inc.campaign_id) & (fixed["metric"] == metric)
                  & fixed["date"].between(lo, hi)][["date", "route", "notification", "pct_change"]]
        b = b.rename(columns={"route": "fixed_route", "notification": "fixed_notification",
                              "pct_change": "fixed_pct_change"})
        r = f.merge(b, on="date", how="left").sort_values("date")
        r.insert(0, "incident_id", inc.incident_id)
        r.insert(1, "incident_type", inc.incident_type)
        r.insert(2, "incident_severity", inc.severity)
        r["incident_start"] = inc.start_date
        r["incident_end"] = inc.end_date
        r["in_incident_window"] = r["date"].between(inc.start_date, inc.end_date)
        frames.append(r)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_comparison(syn: dict) -> dict:
    """Controlled comparison on the synthetic EVAL split.

    fixed_threshold   - src/baselines.py rule, threshold chosen on DEV only
    adaptive_detector - same flags as full Adomaly, every flag notified
                        (no evidence routing, no episodes)
    full_adomaly      - the shipped pipeline output, unmodified
    """
    metrics, labels, benign = syn["_metrics"], syn["_labels"], syn["_benign"]
    full = syn["_records"]
    threshold, grid = select_fixed_threshold(metrics, labels, benign)
    fixed = fixed_threshold_records(metrics, threshold)
    recs = {"fixed_threshold": fixed, "adaptive_detector": notify_every_flag(full),
            "full_adomaly": full}

    eval_labels = labels[labels["period"] == "eval"]
    rows, cov, rp_rows, rp = [], [], [], {}
    for key in SYSTEMS:
        r = recs[key]
        m = system_metrics(key, r, labels, benign, period="eval")
        m["system_label"] = SYSTEM_LABELS[key]
        rows.append(m)
        c = notification_coverage(r, eval_labels)
        c.insert(0, "system", key)
        cov.append(c)
        lab = label_records(r[r["period"] == "eval"], labels, benign)
        rp[key] = route_precision(lab)
        for route, d in rp[key].items():
            if d["flags"]:
                rp_rows.append({"system": key, "system_label": SYSTEM_LABELS[key],
                                "route": route, **d})

    return {
        "threshold": threshold, "grid": grid,
        "comparison": pd.DataFrame(rows),
        "coverage": pd.concat(cov, ignore_index=True),
        "route_precision": rp, "route_precision_df": pd.DataFrame(rp_rows),
        "replay": build_replay(full, fixed, labels),
    }


def run_ps04_burden(ps04_metrics: pd.DataFrame, ps04_records: pd.DataFrame,
                    threshold: float) -> pd.DataFrame:
    """Notification volume on PS-04 per system. No labels exist, so this is
    volume only; nothing here is an accuracy measure."""
    rows = [
        {"system": "adaptive_detector", **notification_burden(notify_every_flag(ps04_records)),
         "status": "ok", "note": ""},
        {"system": "full_adomaly", **notification_burden(ps04_records), "status": "ok", "note": ""},
    ]
    # The fixed rule on PS-04 reuses the DEV-selected synthetic threshold and the
    # metric set the PS-04 detector scores. It is attempted, and any failure is
    # recorded rather than hidden or worked around.
    try:
        from src.detector_config import PS04_BASE_METRICS, RATIO_METRICS, DetectorConfig
        cfg = DetectorConfig(metrics=PS04_BASE_METRICS + tuple(RATIO_METRICS))
        cols = ["date", "campaign_id", *PS04_BASE_METRICS]
        fx = fixed_threshold_records(ps04_metrics[cols], threshold, cfg, allowed_columns=cols)
        rows.insert(0, {"system": "fixed_threshold", **notification_burden(fx),
                        "status": "ok", "note": f"threshold {threshold:.0%} chosen on synthetic DEV"})
    except Exception as e:  # recorded honestly in the CSV and report
        rows.insert(0, {"system": "fixed_threshold", "raw_flagged_records": np.nan,
                        "notifications_sent": np.nan, "suppressed_repeats": np.nan,
                        "notification_reduction_pct": np.nan, "status": "not_run",
                        "note": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    df.insert(1, "system_label", df["system"].map(SYSTEM_LABELS))
    df.insert(0, "dataset", "PS-04 (official, no labels)")
    return df


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def write_comparison_outputs(cmp: dict, ps04_burden: pd.DataFrame):
    """Writes the Round-2 files and appends sections 8-9 to report.md
    (sections 1-7, written by write_outputs, are left exactly as they were)."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comp = cmp["comparison"]
    lead = ["system", "system_label"]
    comp = comp[lead + [c for c in comp.columns if c not in lead]]
    comp.to_csv(REPORT_DIR / "system_comparison.csv", index=False)
    cmp["grid"].to_csv(REPORT_DIR / "fixed_threshold_grid.csv", index=False)
    cmp["coverage"].to_csv(REPORT_DIR / "incident_notification_coverage.csv", index=False)
    cmp["route_precision_df"].to_csv(REPORT_DIR / "route_precision.csv", index=False)
    cmp["replay"].to_csv(REPORT_DIR / "synthetic_incident_replay.csv", index=False)
    ps04_burden.to_csv(REPORT_DIR / "ps04_notification_burden.csv", index=False)

    (REPORT_DIR / "system_comparison.json").write_text(json.dumps({
        "note": "Synthetic benchmark, EVAL split only. Designed scenarios, not "
                "real-world performance. Evidence scores are heuristic, not probabilities.",
        "fixed_threshold": {
            "rule": "flag if |value(t) / value(t-1) - 1| >= threshold",
            "selected_threshold": cmp["threshold"],
            "selection": "max record-level F1 on DEV only; ties -> largest threshold",
            "grid": cmp["grid"].to_dict(orient="records"),
        },
        "systems": comp.to_dict(orient="records"),
        "route_precision_eval": cmp["route_precision"],
    }, indent=2, default=_json_default))

    # ---- report.md, appended ----
    def pct(x):
        return _fmt_pct(x)

    def num(x, nd=0):
        return "n/a" if pd.isna(x) else f"{x:.{nd}f}"

    L = ["\n## 8. Controlled comparison — synthetic EVAL split\n\n",
         "Three systems scored on the same records, labels and matching rule as sections 1-3:\n\n",
         f"- **Fixed threshold**: flag when a metric moves by at least "
         f"**{cmp['threshold']:.0%}** vs the previous day (src/baselines.py). The threshold "
         "was chosen by DEV-only F1 over a 20-90% grid (`fixed_threshold_grid.csv`); every "
         "flag is a notification.\n"
         "- **Adaptive detector**: the Adomaly detector's flags, but every flag is notified "
         "(no evidence routing, no episode grouping).\n"
         "- **Full Adomaly**: the shipped pipeline output (routing + episodes + escalation).\n\n",
         "An incident counts as **notified** only if a `new` or `escalation` notification "
         "lands on its mapped metric inside its window; suppressed flags do not count.\n\n",
         "| system | precision | recall | F1 | FPR | incidents detected | notified | "
         "missed | median notify delay (d) | raw flags | notifications | reduction |\n",
         "|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for r in comp.itertuples():
        L.append(f"| {r.system_label} | {pct(r.precision)} | {pct(r.recall)} | "
                 f"{num(r.f1, 3)} | {pct(r.false_positive_rate)} | "
                 f"{r.incidents_detected}/{r.n_incidents} | {r.incidents_notified}/{r.n_incidents} | "
                 f"{r.incidents_missed} | {num(r.median_notification_delay_days, 1)} | "
                 f"{r.raw_flagged_records} | {r.notifications_sent} | "
                 f"{_fmt_pct(r.notification_reduction_pct / 100 if pd.notna(r.notification_reduction_pct) else np.nan)} |\n")
    L.append("\nNotification burden counts all scored metrics in the EVAL split; "
             "precision/recall/F1/FPR count only spend, conversions and ctr "
             "(the metrics labels target).\n")
    full = comp.set_index("system").loc["full_adomaly"]
    L.append(f"\nFull Adomaly: {full.incidents_detected_not_notified} incident(s) were "
             "detected but every in-window flag was suppressed by an earlier episode, and "
             f"{full.incidents_reaching_human_via_review_only} reached a human only through "
             "Human Review. Per-incident detail: `incident_notification_coverage.csv`.\n")

    L.append("\n**Route precision (EVAL, labelled metrics only)**\n\n"
             "| system | route | flags | TP | FP | precision |\n|---|---|---|---|---|---|\n")
    for r in cmp["route_precision_df"].itertuples():
        L.append(f"| {r.system_label} | {r.route} | {r.flags} | {r.tp} | {r.fp} | "
                 f"{pct(r.precision)} |\n")
    L.append("\nIf Alert precision is not clearly above Human Review precision, the "
             "evidence score is not separating cases on this benchmark; that is reported "
             "as-is, not adjusted.\n")

    L.append("\n## 9. PS-04 notification burden by system (volume only)\n\n"
             "PS-04 has no labels. These are counts of notifications each system would send, "
             "not accuracy.\n\n| system | raw flags | notifications | reduction | status |\n"
             "|---|---|---|---|---|\n")
    for r in ps04_burden.itertuples():
        L.append(f"| {r.system_label} | {num(r.raw_flagged_records)} | "
                 f"{num(r.notifications_sent)} | "
                 f"{_fmt_pct(r.notification_reduction_pct / 100 if pd.notna(r.notification_reduction_pct) else np.nan)} | "
                 f"{r.status}{(' — ' + r.note) if r.note else ''} |\n")

    with open(REPORT_DIR / "report.md", "a", encoding="utf-8") as fh:
        fh.write("".join(L))


def write_outputs(syn: dict, ps04: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rl = syn["record_level"]

    # ---- JSON ----
    (REPORT_DIR / "synthetic_metrics.json").write_text(json.dumps({
        "evaluation_unit": "(campaign_id, metric, date), metric in "
                           "{spend, conversions, ctr}, EVAL split only",
        "record_level": vars(rl),
        "incident_level": {
            "n_eval_incidents": syn["n_eval_incidents"],
            "detection_rate_eval": syn["detection_rate_eval"],
            "mean_delay_days_eval": syn["mean_delay_days_eval"],
            "median_delay_days_eval": syn["median_delay_days_eval"],
        },
        "false_alerts_on_labelled_benign_shifts_eval": syn["false_alerts_eval"],
        "alert_fatigue_eval_split": syn["fatigue_eval"],
        "alert_fatigue_full_dataset": syn["fatigue_full"],
    }, indent=2, default=str))

    syn["incident_results_all"].to_csv(REPORT_DIR / "synthetic_incident_results.csv", index=False)
    (REPORT_DIR / "synthetic_false_alerts.json").write_text(
        json.dumps(syn["false_alerts_eval"], indent=2, default=str))

    (REPORT_DIR / "ps04_metrics.json").write_text(json.dumps({
        "note": "PS-04 has no ground-truth labels. Precision, recall, F1 and "
                "incident-level detection rate are NOT computed and MUST NOT be "
                "inferred from this file. Only alert-fatigue volume is measured.",
        **ps04,
    }, indent=2, default=str))

    comp = pd.DataFrame([
        {"dataset": "Synthetic (eval split)", **syn["fatigue_eval"]},
        {"dataset": "Synthetic (full)", **syn["fatigue_full"]},
        {"dataset": "PS-04 (official, no labels)", **ps04["fatigue"]},
    ])
    comp.to_csv(REPORT_DIR / "alert_fatigue_comparison.csv", index=False)

    # ---- Markdown ----
    inc_eval = syn["incident_results_eval"]
    lines = []
    lines.append("# Adomaly — Evaluation Report\n")
    lines.append("Generated by `scripts/run_evaluation.py` against the existing detection "
                 "pipeline (`src/pipeline.py`) with its shipped default thresholds "
                 "(`src/detector_config.py: DetectorConfig()`), unmodified.\n")

    lines.append("\n## 1. Matching rule and evaluation unit\n")
    lines.append(
        "- Evaluation unit: `(campaign_id, metric, date)`.\n"
        "- An incident's `affected_metric` is mapped to a detector metric: "
        "`spend->spend`, `conversions->conversions`, `ctr->ctr`, "
        "`conversion_rate->conversions` (the same mapping already used by "
        "`tests/test_detection.py::test_dev_benchmark_incidents_are_flagged`).\n"
        "- A record is **flagged** if its route is `Alert` or `Human Review` "
        "(both reach a human; `No Alert` does not).\n"
        "- A record **detects** an incident if it is flagged, its metric matches the "
        "incident's mapped metric, its campaign matches, and its date falls inside "
        "`[start_date, end_date]`.\n")

    lines.append("\n## 2. Synthetic benchmark — record-level precision / recall / F1\n")
    lines.append(
        "**Split:** EVAL only (chronologically after `DEV_END`). Thresholds in "
        "`DetectorConfig()` were chosen on the DEV period only "
        "(see comment in `src/detector_config.py`), so reporting on DEV would score "
        "the detector on its own tuning set. All numbers below are EVAL-only.\n\n"
        "**Why P/R/F1 are valid here:** the synthetic generator's incident and "
        "benign-event lists are exhaustive — every injected anomalous or shifted "
        "period is labelled — so a `(campaign, metric, date)` outside a labelled "
        "incident window is a genuine negative, not an unlabeled unknown. The "
        "universe is restricted to `metric in {spend, conversions, ctr}` because no "
        "label targets `impressions`, `clicks`, `cvr` or `revenue`; flags on those "
        "are reported separately as informational only, never as false positives.\n\n")
    lines.append(f"- Evaluated records: **{rl.n_evaluated}** "
                 f"(positive: {rl.n_positive}, negative: {rl.n_negative})\n"
                 f"- TP={rl.tp}  FP={rl.fp}  FN={rl.fn}  TN={rl.tn}\n"
                 f"- **Precision: {_fmt_pct(rl.precision)}**\n"
                 f"- **Recall: {_fmt_pct(rl.recall)}**\n"
                 f"- **F1: {_fmt(round(rl.f1, 3) if pd.notna(rl.f1) else rl.f1)}**\n"
                 f"- Flags on unmapped metrics (informational, not scored): "
                 f"{rl.n_unmapped_metric_flags}\n")

    lines.append("\n## 3. Incident-level detection rate and delay (EVAL split)\n")
    lines.append(f"- Incidents in EVAL split: **{syn['n_eval_incidents']}**\n"
                 f"- **Detection rate: {_fmt_pct(syn['detection_rate_eval'])}** "
                 f"({int(inc_eval['detected'].sum())}/{len(inc_eval)} incidents had at least "
                 "one flagged record on the mapped metric inside their window)\n"
                 f"- Mean detection delay (detected incidents only): "
                 f"{_fmt(round(syn['mean_delay_days_eval'], 1) if pd.notna(syn['mean_delay_days_eval']) else syn['mean_delay_days_eval'])} days\n"
                 f"- Median detection delay: "
                 f"{_fmt(syn['median_delay_days_eval'])} days\n\n"
                 "Per-incident detail:\n\n")
    lines.append("| incident_id | campaign | type | metric | severity | detected | delay (days) |\n"
                 "|---|---|---|---|---|---|---|\n")
    for r in inc_eval.itertuples():
        lines.append(f"| {r.incident_id} | {r.campaign_id} | {r.incident_type} | "
                     f"{r.mapped_metric} | {r.severity} | {'yes' if r.detected else '**no**'} | "
                     f"{_fmt(round(r.delay_days,0) if pd.notna(r.delay_days) else r.delay_days)} |\n")

    fa = syn["false_alerts_eval"]
    lines.append("\n## 4. False alerts on labelled benign (non-incident) shifts — EVAL split\n\n")
    lines.append(
        "Benign events are deliberate, sustained shifts (e.g. a planned budget increase) "
        "that a good detector should stop re-flagging once past the baseline window.\n\n"
        f"- Benign-shift records in EVAL: {fa['benign_shift_records']}\n"
        f"- Flagged (Alert/Review): {fa['benign_shift_flagged_records']} "
        f"({_fmt_pct(fa['benign_shift_flag_rate'])})\n"
        f"- Of those, routed to Alert: {fa['benign_shift_alert_records']}\n")

    lines.append("\n## 5. Alert-fatigue behaviour\n")
    for label, d in [("Synthetic, EVAL split", syn["fatigue_eval"]),
                     ("Synthetic, full dataset", syn["fatigue_full"]),
                     ("PS-04 (official, no labels)", ps04["fatigue"])]:
        lines.append(f"\n**{label}**\n\n"
                     f"- Raw flagged records (naive per-day alerting): {d['raw_flagged_records']}\n"
                     f"- Alerts: {d['alerts']}  ·  Human Review: {d['human_review_records']}\n"
                     f"- Episodes: {d['episodes']}\n"
                     f"- Notifications actually sent: {d['notifications_sent']} "
                     f"(new: {d['new_episode_notifications']}, escalation: {d['escalation_notifications']})\n"
                     f"- Suppressed repeats: {d['suppressed_repeats']}\n"
                     f"- Notification reduction vs raw flags: {_fmt_pct(d['notification_reduction_pct']/100 if pd.notna(d['notification_reduction_pct']) else d['notification_reduction_pct'])}\n")

    lines.append("\n## 6. PS-04 official dataset\n")
    lines.append(
        f"- Campaigns: {ps04['n_campaigns']}  ·  Scored records: {ps04['n_records']}\n"
        "- **No ground-truth labels exist for this dataset.** Precision, recall, F1 and "
        "incident-level detection rate are not computed and must not be inferred from "
        "this report. Only the alert-fatigue volume numbers above (section 5) apply to it.\n")

    lines.append("\n## 7. Assumptions and limitations\n")
    lines.append(
        "- Detector thresholds are the shipped defaults; this evaluation does not tune them.\n"
        "- Metrics are computed only over `metric in {spend, conversions, ctr}`; "
        "`impressions`, `clicks`, `cvr` and `revenue` flags are not scored against any label.\n"
        "- `conversion_rate` incidents (gradual_degradation) are matched on the `conversions` "
        "metric, not `cvr`, for consistency with the project's own existing test "
        "(`test_dev_benchmark_incidents_are_flagged`); a `cvr` flag on the same incident is "
        "not separately credited or penalized.\n"
        "- Episode objects aren't period-tagged in `src/episodes.py`; the EVAL-split fatigue "
        "numbers approximate this by filtering episodes whose `end_date` falls in or after "
        "the EVAL period, and by scoring fatigue over EVAL-period records directly for the "
        "primary record-count figures.\n"
        "- Detection delay is measured in calendar days from an incident's documented "
        "`start_date`, not from when the detector had enough history — an incident that "
        "starts before a campaign has `min_history` days of data could show as 'not detected' "
        "for a reason unrelated to detector sensitivity. None of the EVAL incidents are this "
        "early in their campaign's history, but this is a theoretical gap in the method.\n"
        "- The synthetic benchmark is a designed scenario set, not a sample of real campaign "
        "incidents; these numbers describe behaviour on that benchmark, not real-world "
        "performance, exactly as `app.py`'s own About tab already states.\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


def main():
    syn = run_synthetic_evaluation()
    ps04, ps04_metrics, ps04_records = run_ps04_evaluation(keep_records=True)
    write_outputs(syn, ps04)
    cmp = run_comparison(syn)
    ps04_burden = run_ps04_burden(ps04_metrics, ps04_records, cmp["threshold"])
    write_comparison_outputs(cmp, ps04_burden)

    rl = syn["record_level"]
    print(f"EVAL split: precision={rl.precision:.3f} recall={rl.recall:.3f} "
          f"f1={rl.f1:.3f} (TP={rl.tp} FP={rl.fp} FN={rl.fn})")
    print(f"Incident detection rate (eval): {syn['detection_rate_eval']:.1%} "
          f"({int(syn['incident_results_eval']['detected'].sum())}/{syn['n_eval_incidents']})")
    print(f"\nFixed threshold selected on DEV: {cmp['threshold']:.0%}")
    cols = ["system", "precision", "recall", "f1", "false_positive_rate",
            "incidents_detected", "incidents_notified", "incidents_missed", "n_incidents",
            "median_notification_delay_days", "raw_flagged_records", "notifications_sent",
            "notification_reduction_pct"]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print("\nSystem comparison (synthetic EVAL split):")
        print(cmp["comparison"][cols].to_string(index=False))
        print("\nRoute precision (EVAL, labelled metrics):")
        print(cmp["route_precision_df"][["system", "route", "flags", "tp", "fp",
                                          "precision"]].to_string(index=False))
        print("\nPS-04 notification burden (volume only, no labels):")
        print(ps04_burden[["system", "raw_flagged_records", "notifications_sent",
                           "notification_reduction_pct", "status"]].to_string(index=False))
    print(f"\nWrote reports to {REPORT_DIR.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
