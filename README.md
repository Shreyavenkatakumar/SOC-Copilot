# SOC Copilot — Phase I

AI-Powered Security Log Anomaly Detection Engine.

## Status

Module 1 (Project Setup and Log Collection) implemented.
Module 2 (Drain3 Log Parsing) implemented.
Module 3 (Feature Extraction + Isolation Forest Detection + Evaluation) implemented.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the LogHub HDFS dataset files into `data/raw/`:

```
data/raw/HDFS.log
data/raw/anomaly_label.csv
data/raw/HDFS.log_templates.csv        # validation only
data/raw/Event_traces.csv              # validation only
data/raw/Event_occurrence_matrix.csv   # validation only
data/raw/HDFS.npz                      # validation only
```

## Run

```bash
python main.py
```

This executes, in order: Module 1 (log collection) → Module 2 (Drain3
parsing) → Module 3a (feature extraction) → Module 3b (Isolation Forest
training + scoring) → Module 3c (evaluation).

### Run only Module 1

```bash
python -m data.log_collector
```

### Run only Module 2

```bash
python -m parsers.parser
```

This streams `data/raw/HDFS.log`, extracts the `<Date> <Time> <Pid>
<Level> <Component>: <Content>` header from every line, mines log
templates from the `Content` field with Drain3, and writes:

- `data/processed/parsed_logs.csv` — one row per raw line, with
  `LineId, Date, Time, Timestamp, Pid, Level, Component, Content,
  BlockId, EventId, EventTemplate, ParameterList, RawLine`
- `outputs/models/drain3_state.bin` — trained Drain3 tree snapshot,
  reloaded automatically on the next run instead of retraining
- `outputs/reports/parsing_report.json` — total lines processed, unique
  template count, most frequent templates, and parsing success rate

On the full LogHub HDFS dataset (11,175,629 lines) this mines 48 unique
templates with a 100% parsing success rate in roughly 8–9 minutes on a
single CPU core.

### Run only Module 3 (requires `data/processed/parsed_logs.csv` from Module 2)

Run all three parts together:

```bash
python -m training.train_isolation_forest
```

Or run each part individually:

```bash
python -m features.feature_extractor          # Part A
python -m models.isolation_forest_model        # Part B
python -m evaluation.evaluate_isolation_forest # Part C
```

**Part A — Feature Extraction** (`features/feature_extractor.py`)
Aggregates `parsed_logs.csv` by `BlockId` into one feature vector per
block, then min-max scales every numeric column to `[0, 1]`:

- `total_log_events`, `event_sequence_length` — log lines in the block
- `unique_template_count` — distinct Drain3 EventIds seen
- `tmpl_E<n>_count` — one column per EventId (template frequency
  distribution), zero-filled for templates absent from a block
- `time_duration_seconds` — span between the block's first and last
  timestamp
- `event_density` — `total_log_events / (time_duration_seconds + ε)`
- `unique_component_count` — distinct Hadoop components involved
- `info_count`, `warn_count`, `error_count`, `other_count` — log level
  counts

Outputs:
- `data/processed/features.csv` — final scaled feature matrix
- `outputs/models/feature_scaler.joblib` — fitted `MinMaxScaler` (reused
  for consistent scaling at inference time / by later modules)
- `outputs/reports/feature_extraction_report.json`

On the full dataset: 575,061 blocks × 58 feature columns.

**Part B — Isolation Forest** (`models/isolation_forest_model.py`)
Trains `sklearn.ensemble.IsolationForest` on an 80% split of
ground-truth **Normal**-only blocks (semi-supervised), then scores
every block in `features.csv`.

Outputs:
- `outputs/models/isolation_forest_model.joblib`
- `outputs/reports/isolation_forest_predictions.csv` — `BlockId,
  iforest_raw_score, iforest_anomaly_score (0-1, higher = more
  anomalous), iforest_prediction (Normal/Anomaly)`

**Part C — Evaluation** (`evaluation/evaluate_isolation_forest.py`)
Joins predictions against `anomaly_label.csv` and computes Accuracy,
Precision, Recall, F1, ROC-AUC, and the confusion matrix.

Output: `outputs/reports/isolation_forest_evaluation.json`

Measured result on the full HDFS dataset (575,061 blocks, contamination
left at scikit-learn's `"auto"` default):

| Metric | Value |
|---|---|
| Accuracy | 0.765 |
| Precision | 0.092 |
| Recall | 0.789 |
| F1 | 0.164 |
| ROC-AUC | 0.882 |

ROC-AUC (0.88) shows the anomaly score itself ranks anomalous blocks
well above normal ones. Precision is low at the default decision
threshold because Isolation Forest's `"auto"` contamination flags far
more blocks than the ~2.9% true anomaly rate in this dataset — this is
expected from a single unsupervised detector and is exactly what the
fusion engine (Isolation Forest + Autoencoder) and severity scoring in
later modules are designed to correct.

## Project Structure

```
SOC_Copilot_Phase1/
├── data/
│   ├── raw/                        # place LogHub HDFS files here
│   ├── processed/
│   │   ├── parsed_logs.csv         # Module 2 output
│   │   └── features.csv            # Module 3a output
│   └── log_collector.py            # Module 1
├── parsers/
│   ├── drain3_config.ini           # Drain3 tuning + masking rules
│   ├── log_header_parser.py        # HDFS header/content/BlockId extraction
│   └── parser.py                   # Module 2
├── features/
│   └── feature_extractor.py        # Module 3a
├── models/
│   └── isolation_forest_model.py   # Module 3b
├── training/
│   └── train_isolation_forest.py   # Module 3 orchestrator (3a + 3b + 3c)
├── evaluation/
│   └── evaluate_isolation_forest.py # Module 3c
├── dashboard/                      # Module 10
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── exceptions.py
├── outputs/
│   ├── logs/
│   ├── reports/
│   │   ├── log_collection_report.json
│   │   ├── parsing_report.json
│   │   ├── feature_extraction_report.json
│   │   ├── isolation_forest_predictions.csv
│   │   └── isolation_forest_evaluation.json
│   ├── models/
│   │   ├── drain3_state.bin           # trained Drain3 tree
│   │   ├── feature_scaler.joblib      # fitted MinMaxScaler
│   │   └── isolation_forest_model.joblib
│   └── figures/
├── main.py
└── requirements.txt
```
