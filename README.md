# Adomaly

Hackathon prototype for PS-04: Campaign Anomaly Alerting Without Alert Fatigue. Adomaly monitors daily advertising campaign metrics, flags anomalies against each campaign's own history, routes uncertain cases to human review, groups repeated alerts into incident episodes, and evaluates its own detection quality against a labelled benchmark.

**Status:** Synthetic benchmark, detection engine (incl. PS-04 dataset support), Streamlit dashboard, and reproducible evaluation module built.

## Synthetic-data limitation

No official dataset was available, so Adomaly uses a deterministic synthetic benchmark (seed 42). Any results reported on it show how the system behaves on designed scenarios. They are not evidence of real-world advertising performance.

## Dataset

15 campaigns × 180 days (2026-01-01 to 2026-06-29), columns `date`, `campaign_id`, `spend`, `impressions`, `clicks`, `conversions`.

- Size tiers: C01–C05 small, C06–C10 medium, C11–C15 large. Small campaigns have few conversions per day, so they are naturally noisy.
- Counts come from latent CPM, CTR and CVR rates, so `conversions ≤ clicks ≤ impressions` always holds. Each campaign has its own weekend effect, slow trend and noise level.
- 12 labelled incidents: spend spike, conversion-tracking failure, CTR collapse and gradual performance degradation. Each type appears in both periods.
- 2 benign events (C06, C10): sustained seasonal budget increases. These are not incidents, and a good detector should not keep alerting on them.

| File | Purpose | Used by detector? |
|---|---|---|
| `data/raw/campaign_metrics.csv` | Daily metrics | Yes |
| `data/raw/incident_labels.csv` | Ground-truth incidents | No, evaluation only |
| `data/raw/benign_events.csv` | Known non-incident shifts | No, evaluation only |

`src/data_io.load_metrics()` rejects any metrics file containing extra columns, so labels cannot leak into detector input.

## Chronological split

| Period | Dates | Use |
|---|---|---|
| Warm-up | 2026-01-01 to 2026-01-28 | No incidents, so baselines have clean history |
| Development / calibration | 2026-01-01 to 2026-03-31 (days 0–89) | Tune thresholds and policy |
| Held-out evaluation | 2026-04-01 to 2026-06-29 (days 90–179) | Report results only |

No incident crosses the boundary. Detectors may use metric history up to day t (including dev-period metrics when scoring eval days), but never future days or labels.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_generation     # writes data/raw/*.csv (synthetic benchmark)
python -m pytest                  # runs tests
python -m scripts.run_evaluation  # writes reports/evaluation/ (precision/recall/F1, detection rate, alert fatigue)
streamlit run app.py              # dashboard (needs data/raw/daily_campaign_metrics.csv)
```

## Dashboard

`streamlit run app.py` loads the organizer-supplied PS-04 file `data/raw/daily_campaign_metrics.csv` and runs `src.pipeline.run_ps04_detection` on it (cached). Tabs: Overview, Campaign explorer, Anomalies, Alert routing, Alert fatigue, **Evaluation**, About & limitations; flagged records, routes and episodes export to CSV. Sidebar filters (channel, vertical, campaign, dates) change the view only; detection always runs on the full history. The PS-04 data has no ground-truth labels, so flags are signals to investigate, not confirmed incidents, and evidence strength is a heuristic, not a probability.

The **Evaluation** tab displays the last saved report from `reports/evaluation/` — it reads the saved files and never re-runs the evaluation itself. Run `python -m scripts.run_evaluation` first (and again after any detector change) to refresh it.

## Evaluation

Reproducible evaluation lives in `src/evaluation.py` (matching rule + metrics) and `scripts/run_evaluation.py` (orchestration + report writer). It runs the existing pipeline with its shipped default thresholds — it does not tune anything.

```bash
python -m scripts.run_evaluation
```

writes to `reports/evaluation/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable report: definitions, numbers, assumptions, limitations |
| `synthetic_metrics.json` | Record-level precision/recall/F1 + incident-level detection rate/delay (EVAL split) |
| `synthetic_incident_results.csv` | One row per labelled incident: detected? delay in days? |
| `synthetic_false_alerts.json` | Flags raised on labelled benign (non-incident) shifts |
| `ps04_metrics.json` | Alert-fatigue volume only — **no accuracy metrics**, see below |
| `alert_fatigue_comparison.csv` | Raw flags vs. notifications sent, synthetic vs. PS-04 |

**What is and isn't measured**

- **Synthetic benchmark** has exhaustive, documented ground truth, so precision, recall, F1, incident-level detection rate and detection delay are computed — but **only on the EVAL split**, since thresholds in `DetectorConfig()` were chosen on DEV. Reporting on DEV would score the detector on its own tuning set.
- **PS-04 official dataset** has no ground-truth labels at all. Precision, recall, F1 and detection rate are **not computed and must not be inferred** for it. Only alert-fatigue volume (raw flags vs. notifications sent after episode suppression) is reported, since that needs no labels.

**Matching rule:** an incident's `affected_metric` maps to one detector metric —

| affected_metric | detector metric |
|---|---|
| `spend` | `spend` |
| `conversions` | `conversions` |
| `ctr` | `ctr` |
| `conversion_rate` | `conversions` |

(same mapping already used by `tests/test_detection.py::test_dev_benchmark_incidents_are_flagged`). A record is **flagged** if its route is `Alert` or `Human Review`. A record **detects** an incident if it is flagged, on the mapped metric, for the right campaign, inside `[start_date, end_date]`. Precision/recall are computed only over `metric in {spend, conversions, ctr}` — the only metrics any label targets; flags on `impressions`, `clicks`, `cvr` or `revenue` are reported separately as informational, never as false positives.

`tests/test_evaluation.py` covers the matching rule, precision/recall/F1 edge cases (perfect detection, FP/FN counting, unmapped-metric exclusion, 0/0 → NaN not a crash), incident-level detection/delay, benign false-alert counting, alert-fatigue summarization, and an end-to-end smoke test against the real pipeline and benchmark data.

## Layout

```
app.py                      Streamlit dashboard (PS-04 dataset + Evaluation tab)
src/config.py                Seed, dates, split, file paths
src/data_generation.py       Synthetic benchmark generator
src/data_io.py                Loaders that keep metrics and labels separate
src/evaluation.py             Matching rule, precision/recall/F1, incident detection, alert fatigue
scripts/run_evaluation.py     Runs the pipeline and writes reports/evaluation/
tests/                        pytest suite
reports/evaluation/           Generated evaluation report (md/json/csv)
data/raw/  data/processed/  outputs/
```
