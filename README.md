# SOC Copilot — Phase I

AI-powered HDFS log anomaly detection and SOC analyst investigation dashboard.

## Current status

This repository contains the final Phase I architecture:

```text
HDFS logs → Log Collection → Drain3 → 58 features
                         ├→ Autoencoder (58)
                         └→ IF-v2 (19)
                              ↓
Validation-only isotonic calibration
                              ↓
0.10 IF-v2 + 0.90 Autoencoder score fusion
                              ↓
Production threshold = 0.323459
                              ↓
Normal / Anomaly → Severity → SHAP → Analyst dashboard
```

The legacy `models/isolation_forest_model.py` is retained only as reference/rollback code. It is not imported or executed by the production pipeline.

## Dataset

- 11,175,629 HDFS log lines
- 575,061 labelled HDFS blocks
- 558,223 Normal blocks
- 16,838 Anomaly blocks
- 48 Drain3 event templates
- 58 engineered feature columns

Detector training is Normal-only. The final split is:

| Split | Blocks | Normal | Anomaly |
|---|---:|---:|---:|
| Training | 446,578 | 446,578 | 0 |
| Validation | 64,241 | 55,822 | 8,419 |
| Test | 64,242 | 55,823 | 8,419 |

Block IDs are checked for overlap across splits.

## Models

### Autoencoder

All 58 features are used. Architecture:

```text
58 → 32 → 16 → 8 → 16 → 32 → 58
```

The Autoencoder is trained using Normal-only training blocks and uses reconstruction error as anomaly evidence.

### Isolation Forest v2

IF-v2 is the only Isolation Forest used in production.

```text
n_estimators = 200
max_samples = 0.5
max_features = 1.0
bootstrap = False
random_state = 42
n_jobs = -1
```

It uses 19 selected features. Feature selection and the IF-v2 scaler are derived from training data only; the decision threshold is selected on validation data only.

### Fusion

The production fusion configuration is frozen:

```text
fusion_score = 0.10 × calibrated_IFv2_score
             + 0.90 × calibrated_AE_score
```

```text
fusion_score >= 0.323459 → Anomaly
fusion_score <  0.323459 → Normal
```

Both calibrators are fitted on validation data only. The production path does not perform a new fusion grid search.

## Explainability

The dashboard provides SHAP explanations for the actual production scoring functions:

- Autoencoder reconstruction-error evidence over all 58 features.
- IF-v2 anomaly-score evidence over the persisted 19 features.
- Positive SHAP contribution means stronger anomaly evidence.
- Negative SHAP contribution means weaker anomaly evidence.

SHAP is model evidence for analyst investigation; it is not treated as proof of root cause.

## Dashboard

Run:

```bash
python -m streamlit run dashboard/app.py
```

Five primary pages are provided:

1. SOC Overview
2. Alerts
3. Alert Investigation
4. Analytics
5. Model & System Evidence

The Alert Investigation page connects an alert to its fusion score, model evidence, SHAP contributors, behavioural features, event templates, Block Activity Timeline, and supporting parsed log evidence.

The dashboard is read-only with respect to trained ML artifacts: it does not retrain models, refit scalers/calibrators, change fusion weights, or modify production predictions.

## Reproduction

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The main pipeline executes Log Collection → Drain3 parsing → Feature Extraction → IF-v2 → Autoencoder → Autoencoder evaluation → production fusion → Fusion evaluation.

## Production vs reference code

Production path:

- `main.py`
- `models/autoencoder_model.py`
- `models/isolation_forest_model_v2.py`
- `models/fusion_engine.py`
- `evaluation/evaluate_autoencoder.py`
- `evaluation/evaluate_fusion.py`
- `dashboard/`

Reference/historical material may remain in the repository for traceability. In particular, `models/isolation_forest_model.py` is legacy/reference code and is not part of the production decision path.

## Publication consistency

Source code and output artifacts must correspond to the same final revision when the repository is cited by a paper. Older evaluation reports or experimental reports should not be presented as the final production result.

Before publication, regenerate the final prediction/evaluation artifacts from the final source revision and commit them together. This prevents the source tree, dashboard, metrics, and manuscript from describing different project versions.

## Limitations

This is a Phase I HDFS block-level anomaly detection and investigation system. It does not claim full attack-chain reconstruction, automated root-cause determination, or production SIEM/ticketing integration.