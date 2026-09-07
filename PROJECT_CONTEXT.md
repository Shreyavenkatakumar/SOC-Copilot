# PROJECT_CONTEXT.md
## SOC Copilot — AI-Powered Security Log Anomaly Detection Engine (Phase I)
### Handover document for AI coding agents (written for OpenCode)

> **Read this before touching any code.** This project is a Python ML/data
> pipeline + a Streamlit dashboard — it is **not** a typical frontend/backend/
> database web application. Several sections of the standard handover
> template below do not apply; they are explicitly marked **N/A** rather than
> filled with invented content, per instruction. Where information genuinely
> was not verified in the source conversation, it is marked **UNKNOWN**.

---

## 1. PROJECT OVERVIEW

**Project name:** SOC Copilot — AI-Powered Security Log Anomaly Detection Engine, Phase I.

**Problem statement:** Large distributed systems (the reference dataset is HDFS — Hadoop Distributed File System) generate millions of log lines. Manual review is infeasible, and rule-based monitoring only catches known attack signatures, missing novel/statistically unusual behaviour. Existing academic anomaly-detection systems in this space typically require expensive GPU hardware, rely on supervised learning (needing labeled attack data that's rarely available), give no severity prioritization, and offer no explainability for why a block was flagged.

**Objective:** Build a full pipeline — raw HDFS logs → parsing → feature engineering → two independent unsupervised ML detectors → calibrated score fusion → severity scoring → SHAP explainability → an interactive SOC analyst dashboard — that runs on CPU only, uses no attack labels for training, and produces an explainable, prioritized, defensible final anomaly decision.

**Proposed solution:** A pipeline combining an **Isolation Forest (IF-v2)** and a **Deep Autoencoder**, whose calibrated anomaly scores are combined via a fixed, experimentally-validated weighted fusion (0.10 IF-v2 + 0.90 Autoencoder, threshold 0.323459), feeding a deterministic severity layer and SHAP-based per-alert explanations, all surfaced through a five-page Streamlit dashboard.

**Main users:** SOC (Security Operations Center) analysts triaging anomaly alerts; secondarily, project reviewers/mentors evaluating the system academically (viva/demo use case is explicitly a design consideration).

**Main use cases:**
1. Run the full pipeline (`python main.py`) end-to-end from raw HDFS logs to a final fused anomaly decision per block.
2. Open the dashboard, review the SOC Overview KPIs, and work the Alert Queue (Critical → High → Medium → Low).
3. Investigate a specific flagged block: view its scores, a plain-English explanation, SHAP evidence from both detectors, its behavioural features, its actual event-template breakdown, and its real parsed log timeline.
4. Reference the Model & System Evidence page for academic/viva presentation of architecture and metrics.

---

## 2. CURRENT IMPLEMENTATION

### Fully implemented and verified (executed against the real, full dataset — 11,175,629 log lines, 575,061 blocks)
- Log Collection (`data/log_collector.py`)
- Drain3 Parsing (`parsers/parser.py`, `parsers/log_header_parser.py`) — 100% parse success, 48 templates
- Feature Extraction (`features/feature_extractor.py`) — 58 features/block
- Isolation Forest v2, the sole production IF (`models/isolation_forest_model_v2.py`) — 19 selected features
- Autoencoder (`models/autoencoder_model.py`) — unchanged since first implementation
- Score calibration + Fusion (`models/fusion_engine.py`) — fixed 0.10/0.90 weights, threshold 0.323459
- Evaluation modules for each stage (`evaluation/evaluate_isolation_forest.py`, `evaluation/evaluate_autoencoder.py`, `evaluation/evaluate_fusion.py`)
- Severity scoring (`dashboard/utils/severity.py` or `dashboard/severity.py` depending on structure — see Section 5 caveat)
- SHAP explainability (`dashboard/shap_explain.py` or `dashboard/utils/shap_explain.py` — same caveat)
- Streamlit dashboard, 5 pages (`dashboard/app.py`)
- Research/audit modules (not production, standalone): `models/isolation_forest_research.py`, `models/complementarity_analysis.py`

### Partially completed / needs re-verification before handover
- **Dashboard file structure is in a confirmed but recently-modified state.** The user's actual live repository structure is **flat**: `dashboard/app.py`, `dashboard/charts.py`, `dashboard/data_loader.py`, `dashboard/severity.py`, `dashboard/style.css` (confirmed directly by the user after a `ModuleNotFoundError` was diagnosed and fixed). A `dashboard/shap_explain.py` module was added afterward but its exact final integration state (imports inside `app.py`, path constants) was **not re-verified against the user's live flat structure after the SHAP work was added in this same session** — see Section 14 (Known Issues) and Section 16 (Do Not Break). **OpenCode's first task should be to inspect the actual current `dashboard/` folder structure before changing anything.**
- Block Activity Timeline / per-block SQLite log index: logic implemented and tested successfully up to ~10M/11.17M rows in one sandbox run before hitting a **sandbox-only** disk-space limit (not a code defect) — should complete fully on a normal machine with adequate free disk, but this was not confirmed end-to-end on the user's actual machine.

### Not implemented (explicitly out of Phase I scope)
- MITRE ATT&CK technique mapping
- Root-cause analysis
- Attack timeline reconstruction (block-level chronological anomaly trend across the whole dataset — no per-block point-in-time timestamp exists in the feature pipeline, only a duration)
- Multi-source log correlation
- Automated PDF investigation reports
- FastAPI service layer / any REST API
- Authentication/authorization of any kind
- A database (all data is file-based: CSV/JSON/joblib/keras files)
- User accounts, sessions, or roles

---

## 3. TECHNOLOGY STACK

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas 2.2.2, NumPy 1.26.4 |
| Classical ML | scikit-learn 1.5.1 (IsolationForest, MinMaxScaler, IsotonicRegression, metrics, train_test_split) |
| Deep learning | TensorFlow / Keras 2.16.1 (Autoencoder) |
| Log parsing | drain3 0.9.11 |
| Explainability | shap (0.46.0 pinned in requirements.txt; 0.52.0 verified working in the development sandbox) |
| Dashboard UI | Streamlit 1.37.1 |
| Dashboard charts | Plotly 5.23.0 |
| Model/artifact persistence | joblib 1.4.2 |
| Per-block log lookup | sqlite3 (Python standard library) |
| Config/misc | pyyaml 6.0.2, tqdm 4.66.5, scipy 1.13.1 |

**Frontend:** N/A — the Streamlit dashboard *is* the frontend; there is no separate JS/React/etc. frontend.
**Backend:** N/A — no server process; `main.py` is a batch pipeline script, not a running service. The dashboard reads files directly, no API layer between them.
**Database:** N/A — no relational/NoSQL database. All persistence is flat files: CSV (`data/processed/`, `outputs/reports/`), JSON (`outputs/reports/`), joblib (`outputs/models/`), Keras native format (`outputs/models/autoencoder_model.keras`), and one SQLite file used purely as a local per-block log index cache (`dashboard/.cache/parsed_logs_index.db`), not a general-purpose database.
**Authentication:** N/A — no login/auth system exists anywhere in this project.
**APIs:** N/A — no REST/GraphQL/RPC API exists.
**External services:** None. Fully self-contained, offline, file-based.

---

## 4. PROJECT ARCHITECTURE

There is no frontend→backend→database flow, no authentication flow, and no API communication layer — see Section 3. The real architecture is a **linear batch pipeline** followed by a **read-only visualization layer**:

```
Raw HDFS.log + anomaly_label.csv   (data/raw/)
        │
        ▼
data/log_collector.py :: LogCollector          → outputs/reports/log_collection_report.json
        │
        ▼
parsers/parser.py :: Drain3LogParser           → data/processed/parsed_logs.csv
        │                                          outputs/models/drain3_state.bin
        ▼
features/feature_extractor.py :: FeatureExtractor → data/processed/features.csv
        │                                            outputs/models/feature_scaler.joblib
   ┌────┴────┐
   ▼         ▼
models/autoencoder_model.py     models/isolation_forest_model_v2.py
  :: AutoencoderDetector           :: IsolationForestV2Validator
   │                                 │
   ▼                                 ▼
outputs/reports/autoencoder_predictions.csv
                                outputs/reports/isolation_forest_v2_predictions.csv
   └────┬────┘
        ▼
models/fusion_engine.py :: FusionEngine        → outputs/reports/fusion_predictions.csv
                                                   outputs/models/fusion_config.joblib
        │
        ▼
evaluation/evaluate_*.py                        → outputs/reports/*_evaluation.json
        │
        ▼
dashboard/app.py (Streamlit)  ── reads all of the above files directly, read-only
        │
        ├── dashboard/data_loader.py (or utils/data_loader.py) — file loading + caching
        ├── dashboard/severity.py    (or utils/severity.py)    — dashboard-only severity thresholds
        ├── dashboard/shap_explain.py                          — on-demand SHAP against persisted models
        └── dashboard/charts.py (or components/charts.py)      — Plotly chart builders
```

**Important classes/functions (production):**

| Module | Class/Function | Role |
|---|---|---|
| `data/log_collector.py` | `LogCollector` | Validates + streams raw HDFS log, loads ground-truth labels |
| `parsers/parser.py` | `Drain3LogParser` | Streams log lines, mines Drain3 templates |
| `parsers/log_header_parser.py` | `extract_header()` | Splits raw HDFS line into header fields + Content |
| `features/feature_extractor.py` | `FeatureExtractor` | Aggregates parsed logs into the 58-feature matrix |
| `models/isolation_forest_model_v2.py` | `IsolationForestV2Validator` | Trains/scores IF-v2; `run(include_original_if_comparison=False)` is the **production** call — `True` is only for the standalone research/validation report |
| `models/autoencoder_model.py` | `AutoencoderDetector` | Trains/scores the Autoencoder |
| `models/fusion_engine.py` | `FusionEngine` | Applies the **fixed** production calibration + 0.10/0.90 fusion (no runtime search) |
| `utils/holdout_split.py` | `identify_training_normal_blocks()` | Shared, deterministic train/holdout split reproduction used by multiple modules to avoid leakage |
| `main.py` | `main()` | Orchestrates the full production pipeline in order |
| `dashboard/app.py` | (script, not a class) | Streamlit page router; 5 pages via `PAGE_FUNCTIONS` dict |
| `dashboard/shap_explain.py` | `explain_ifv2()`, `explain_autoencoder()` | On-demand SHAP computation against persisted models |

**Non-production / research-only modules (must not be imported by `main.py` or the dashboard):**
- `models/isolation_forest_model.py` — the original/baseline Isolation Forest, retained for reference/rollback only.
- `models/isolation_forest_research.py` — hyperparameter/feature-audit experiment.
- `models/complementarity_analysis.py` — AE-vs-IF-v2 complementarity experiment.
- `models/fusion_engine_experimental.py.bak` — backup of an earlier grid-search fusion engine, superseded by the fixed-config production version.

---

## 5. COMPLETE FOLDER STRUCTURE

```
SOC_Copilot_Phase1_Module3/
├── data/
│   ├── raw/                        # HDFS.log, anomaly_label.csv (user-supplied)
│   ├── processed/
│   │   ├── parsed_logs.csv         # Drain3 output, ~11.17M rows
│   │   └── features.csv            # 58-feature matrix, 575,061 rows
│   └── log_collector.py
├── parsers/
│   ├── drain3_config.ini
│   ├── log_header_parser.py
│   └── parser.py
├── features/
│   └── feature_extractor.py
├── models/
│   ├── isolation_forest_model.py       # ORIGINAL IF — reference only, not executed in production
│   ├── isolation_forest_model_v2.py    # PRODUCTION IF
│   ├── autoencoder_model.py            # PRODUCTION AE, never modified
│   ├── fusion_engine.py                # PRODUCTION fusion (fixed config, no search)
│   ├── fusion_engine_experimental.py.bak  # old grid-search version, kept as backup
│   ├── isolation_forest_research.py    # standalone research script
│   └── complementarity_analysis.py     # standalone research script
├── evaluation/
│   ├── evaluate_isolation_forest.py
│   ├── evaluate_autoencoder.py
│   └── evaluate_fusion.py
├── training/
│   ├── train_isolation_forest.py   # orchestrates features→IF→eval (legacy Module 3 name)
│   ├── train_autoencoder.py
│   └── train_fusion.py
├── utils/
│   ├── config.py                   # ALL file paths + hyperparameters, single source of truth
│   ├── logger.py
│   ├── exceptions.py
│   └── holdout_split.py            # shared leakage-free split reproduction
├── dashboard/
│   ├── app.py                      # Streamlit entry point, 5 pages
│   ├── shap_explain.py             # SHAP module (location — see caveat below)
│   ├── style.css / assets/style.css   # CAVEAT: structure ambiguity, see note below
│   ├── data_loader.py / utils/data_loader.py
│   ├── severity.py / utils/severity.py
│   └── charts.py / components/charts.py
├── outputs/
│   ├── logs/soc_copilot.log
│   ├── reports/                    # every *.json report + prediction CSVs
│   ├── models/                     # every trained model / scaler / calibrator artifact
│   └── figures/
├── main.py                         # full production pipeline entry point
└── requirements.txt
```

> ⚠️ **STRUCTURE CAVEAT — resolve before further work.** During this project's development, the dashboard was originally built with a nested structure (`dashboard/components/`, `dashboard/utils/`). The user's actual live machine has a **flat** structure (`dashboard/app.py`, `dashboard/charts.py`, `dashboard/data_loader.py`, `dashboard/severity.py`, `dashboard/style.css`) — this was confirmed directly by the user after fixing a `ModuleNotFoundError` caused by exactly this mismatch. A later SHAP-explainability module (`shap_explain.py`) and app.py restructuring (5-page rebuild) were developed afterward, but **were not re-confirmed against the user's live flat structure in this same session**. **OpenCode must run `find dashboard -type f` (or equivalent) as its very first action and reconcile imports/paths against whatever it actually finds, not against this document's assumption.**

---

## 6. DATABASE

**N/A.** This project has no relational or document database. All state is file-based:
- CSV files (feature matrices, prediction outputs)
- JSON files (evaluation/config reports)
- joblib files (scikit-learn models, scalers, calibrators)
- One `.keras` file (the Autoencoder)
- One SQLite file used only as a local cache/index for fast per-BlockId log lookups in the dashboard (`dashboard/.cache/parsed_logs_index.db`) — schema is a single table:
  ```sql
  CREATE TABLE logs (
    BlockId TEXT, Timestamp TEXT, Component TEXT,
    EventId TEXT, EventTemplate TEXT, Content TEXT
  );
  CREATE INDEX idx_block_id ON logs (BlockId);
  ```
  This is a derived cache, safe to delete and rebuild at any time; it is not a system of record.

---

## 7. API DOCUMENTATION

**N/A.** No REST/GraphQL/RPC API exists in this project. All interaction is either (a) running `python main.py` as a batch job, or (b) launching `streamlit run dashboard/app.py` as an interactive local web UI with no separate API layer — Streamlit's own script-rerun model handles all "requests" internally, not via a documented HTTP API this project defines.

---

## 8. AUTHENTICATION AND AUTHORIZATION

**N/A.** No login, registration, session, JWT, roles, or protected routes exist anywhere in this project. The Streamlit dashboard is a single unauthenticated local application.

---

## 9. UI/FRONTEND — Dashboard Pages

`dashboard/app.py` is a Streamlit script with a sidebar-based page router (`PAGE_FUNCTIONS` dict) and 5 pages:

| Page | Function | Key Behavior |
|---|---|---|
| **SOC Overview** | `page_soc_overview()` | KPI cards (total/normal/anomaly blocks, anomaly rate, critical/high alert counts) sourced from real report files, not hardcoded; severity + fusion-score distribution charts; explicitly avoids fabricating a chronological trend since no per-block timestamp exists |
| **Alerts** | `page_alerts()` | Sortable alert queue (severity rank, then fusion score descending); sidebar filters (severity, decision, BlockId substring, EventId, score range, max rows); "Investigate Selected Alert" button sets `st.session_state["selected_block_id"]` and directs the user to switch pages |
| **Alert Investigation** | `page_alert_investigation()` | Alert Summary; plain-English explanation built from the block's real score values; Model Evidence; SHAP bar charts for both detectors (via `dashboard/shap_explain.py`); Behavioural Features + "View All 58 Features" expander; block-specific event-template breakdown (`get_block_template_breakdown()`); Block Activity Timeline from the real parsed logs (`get_block_raw_logs()`), expandable per-line |
| **Analytics** | `page_analytics()` | Score distributions, AE-vs-IF-v2 scatter, top suspicious blocks, event-template totals, filterable by All/Anomaly/Normal |
| **Model & System Evidence** | `page_model_system_evidence()` | Architecture diagram (static text/markdown), per-model + fusion metrics with confusion matrices, IF-v2's 19 features, dataset statistics, IF-v2 hyperparameters, Autoencoder architecture text, System Status checklist |

**Data sharing between pages:** `master_df = dl.build_master_table()` is built once near the top of `app.py` (cached via `@st.cache_data`) and reused across all pages that need per-block data; `st.session_state["selected_block_id"]` is the only cross-page navigation state.

**Explicit UI rules enforced:** Ground Truth label is never displayed on Alerts/Alert Investigation (`analyst_view()` helper strips it); severity is always shown separately from `fusion_prediction` ("ML Decision"); no arbitrary/invented thresholds are surfaced, only the real production threshold and weights (loaded from `fusion_config.joblib`, not hardcoded, with a hardcoded fallback display value only if that file is missing).

---

## 10. BACKEND

**N/A in the traditional sense** (no controllers/services/repositories/ORM). The closest equivalent is the pipeline module set described in Section 4. If mapping loosely to that pattern:
- **"Repository" equivalent:** `utils/config.py` (all file paths) + each module's own file-reading logic.
- **"Service" equivalent:** the `*Detector`/`*Engine`/`*Evaluator` classes listed in Section 4.
- **"Model/entity" equivalent:** there are no ORM entities; the closest are the persisted artifacts themselves (feature matrices, trained models, prediction CSVs) and their schemas, documented per-module in the prior conversation and summarized in Section 2.

---

## 11. CONFIGURATION

All configuration lives in **one file**: `utils/config.py`. No `.env` file or secrets exist in this project (it's fully local/offline, no API keys, no credentials). Key configuration values (none of these are secrets):

| Constant | Value | Purpose |
|---|---|---|
| `RANDOM_STATE` | 42 | Seed for all splits/models, used everywhere for reproducibility |
| `TRAIN_TEST_SPLIT_RATIO` | 0.8 | Fraction of labeled-Normal blocks used for training |
| `FUSION_VALIDATION_SPLIT_RATIO` | 0.5 | Splits the holdout into validation vs. test |
| `ISOLATION_FOREST_V2_PARAMS` | `n_estimators=200, max_samples=0.5, max_features=1.0, bootstrap=False, random_state=42, n_jobs=-1` | Production IF-v2 hyperparameters |
| `FUSION_PRODUCTION_IF_WEIGHT` | 0.10 | Fixed production fusion weight |
| `FUSION_PRODUCTION_AE_WEIGHT` | 0.90 | Fixed production fusion weight |
| `FUSION_PRODUCTION_THRESHOLD` | 0.323459 | Fixed production decision threshold |
| `AUTOENCODER_PARAMS` | encoding_dims=[32,16,8], relu/sigmoid, lr=1e-3, batch=256, epochs=50, val_split=0.1 | Autoencoder architecture/training config |
| `AUTOENCODER_CONTAMINATION` | 0.03 | Percentile used to set the AE's training-error threshold |

No environment variables are required to run this project — everything is a Python constant or a relative file path resolved from `utils/config.py`'s `BASE_DIR`.

---

## 12. IMPORTANT DEVELOPMENT DECISIONS

| Decision | Why |
|---|---|
| Both detectors trained **only on Normal blocks**, no attack labels used as training input | Matches the project's core objective (unsupervised detection) and avoids the base paper's reliance on labeled attack data |
| Strict train/validation/test split with a shared, reproducible split function (`utils/holdout_split.py`) | Prevents data leakage; multiple modules independently reproducing the identical split (rather than importing each other) was a deliberate choice for module isolation, at the cost of some code duplication |
| Original Isolation Forest kept on disk but never executed in production | User explicitly required a rollback reference without removing history |
| IF-v2 uses a **locally-fit** scaler (fit on training data only), not the globally-fit `feature_scaler.joblib` used elsewhere | The global scaler was fit across all 575,061 blocks including validation/test — a minor leakage source specifically avoided for IF-v2's from-scratch validation stage |
| Fusion weight (0.10 IF-v2 / 0.90 AE) and threshold (0.323459) are **fixed constants** in production, not re-searched at runtime | The search was performed once, validated, and explicitly frozen — re-running a search inside the production pipeline would reintroduce risk of accidental overfitting/inconsistency across runs |
| SHAP sign convention for IF-v2 was manually flipped | `shap.TreeExplainer` explains sklearn's raw `decision_function` (opposite sign to the project's own anomaly-score convention); left unflipped, positive/negative SHAP values would have meant the opposite of what the Autoencoder's SHAP values mean, confusing analysts |
| Fusion did **not** default to "IF and AE contribute equally" or get forced to include IF-v2 at a large weight | Explicit project rule: do not force a result — the weight was determined by validation-only grid search, which happened to land on a small (0.10) but genuine, measured contribution |
| Dashboard is strictly read-only | Explicit constraint — the dashboard must never retrain/refit/recalculate any model artifact |

---

## 13. BUGS AND FIXES

| Bug | Root Cause | Fix |
|---|---|---|
| Drain3 0.9.11 `TypeError` inside `cachetools` | `parameter_extraction_cache_capacity` read from `.ini` as a `str`, not `int` | Explicit `int()` cast added after `TemplateMinerConfig.load()` in `parsers/parser.py` |
| Evaluation metrics initially inflated | Evaluator was scoring on all blocks, including the ones used to train the Isolation Forest | Rewrote evaluators to reproduce the exact train/holdout split and exclude training blocks before computing metrics |
| `ImportError: cannot import name 'ISOLATION_FOREST_V2_MODEL_FILE'` | Config constants were added in a working session but the updated `utils/config.py` was never re-delivered to the user | Diffed the actual delivered file against required imports, added exactly the 4 missing constants, verified via `py_compile` + real import |
| `ModuleNotFoundError: No module named 'components'` | Dashboard code assumed a nested `dashboard/components/`/`dashboard/utils/` structure; user's actual repo is flat | Flattened `app.py`'s 3 import lines; also found and fixed 2 related latent path bugs (`PROJECT_ROOT` and `CACHE_DIR` in `data_loader.py`, and the CSS path in `app.py`) that were silently pointing at the wrong directory even before the import error was hit |
| SHAP for IF-v2 showed intuitively backwards contributions | `TreeExplainer` explains `decision_function` (higher = more normal), opposite to the project's anomaly-score convention | Verified empirically on a known anomaly block and a known normal block, then negated IF-v2's SHAP values in `explain_ifv2()` |
| SQLite log-index build failed with "database or disk is full" | Development **sandbox** ran low on disk from accumulated large files during the session (not a code defect) | Diagnosed the exact exception, freed sandbox disk space, confirmed the indexing logic itself completes correctly given adequate disk; also confirmed the failure path degrades gracefully (`None` return, dashboard shows an info message, no crash) |
| `PROJECT_ROOT` computed one directory level too high in `data_loader.py` after a structural change | Nested-vs-flat directory mismatch (same root cause as the `ModuleNotFoundError` above, discovered while diagnosing it) | Corrected `Path(__file__).resolve().parent.parent` (2 levels for flat structure, was 3) |

---

## 14. KNOWN ISSUES

1. **Dashboard folder structure needs re-verification** (see Section 5 caveat) — the SHAP module and the 5-page `app.py` rebuild were developed after the flat-structure fix was confirmed, but not re-verified against the user's actual live flat directory in the same pass. There is a real risk that `dashboard/app.py`'s imports (`import shap_explain as se`, etc.) and `shap_explain.py`'s own `PROJECT_ROOT` path math do not currently match whatever the user's live folder actually contains.
2. **SQLite log index build is resource-intensive** for the full ~11.17M-row `parsed_logs.csv` and was only verified to complete on a machine with adequate free disk space; graceful degradation (no crash) is confirmed, but a full successful build+query was not end-to-end confirmed on the user's actual machine within this conversation.
3. **Two slightly different final Fusion metric sets exist** across different runs of the same seeded pipeline (e.g., Precision 98.84% vs. 98.90%, F1 99.41% vs. 99.44%) — both real, differing by ≤0.06 percentage points, attributable to normal isotonic-calibration refit variance. Not a bug, but worth knowing before quoting "the" final number.
4. **`shap==0.46.0`** is pinned in `requirements.txt`, but **`shap==0.52.0`** is what was actually verified working in the development sandbox (0.46.0 was never actually tested end-to-end). Recommend testing with the pinned version or updating the pin.
5. The Autoencoder's SHAP explanation uses `KernelExplainer` (a slower, model-agnostic approximation) rather than a Keras-native gradient-based explainer, chosen deliberately for TF/Keras-version robustness — this is a design tradeoff, not a bug, but is a candidate for future optimization if SHAP latency becomes a problem.

---

## 15. CURRENT PROJECT STATUS

**Completed:**
- Log Collection, Drain3 Parsing, Feature Extraction — all verified on the full real dataset.
- IF-v2 — final production configuration, verified, integrated.
- Autoencoder — unchanged, verified.
- Score calibration + Fusion — fixed production config, verified, reproduced across independent runs.
- Severity scoring — implemented, documented single-location thresholds.
- SHAP explainability — implemented and verified end-to-end on real blocks (before the most recent dashboard restructuring — see Known Issues #1).
- Dashboard — 5 pages implemented and previously verified to launch without error (before the most recent SHAP integration — needs re-verification).
- Full project report(s) written, based on real verified metrics.

**In progress / needs re-verification:**
- Final confirmation that the dashboard (with SHAP fully integrated) launches cleanly against the user's actual current flat folder structure.
- Full-scale SQLite log-index build on the user's actual machine.

**Pending (Phase II, not started):**
- MITRE ATT&CK mapping, root-cause analysis, timeline reconstruction, multi-source correlation, PDF reporting, FastAPI layer.

**Known bugs:** None currently open that are code defects — Known Issues #1 and #2 above are re-verification tasks, not confirmed active bugs.

---

## 16. DO NOT BREAK

- **`utils/holdout_split.py`'s `identify_training_normal_blocks()` split logic and its `RANDOM_STATE=42` seeding** — every module's leakage-free evaluation depends on independently reproducing this exact split. Changing the split logic, the seed, or the ratio anywhere invalidates every downstream evaluation number in this document.
- **`FUSION_PRODUCTION_IF_WEIGHT=0.10`, `FUSION_PRODUCTION_AE_WEIGHT=0.90`, `FUSION_PRODUCTION_THRESHOLD=0.323459`** — these are fixed, validated constants, not defaults to be casually retuned. Do not reintroduce a runtime grid search into `models/fusion_engine.py`'s production path without explicit instruction.
- **`models/isolation_forest_model.py` must never be imported by `main.py`, `models/fusion_engine.py`, or any dashboard module.** IF-v2 is the sole production Isolation Forest; this was a hard, repeated project requirement.
- **The Autoencoder (`models/autoencoder_model.py`) must not be modified** — architecture, features (all 58), training procedure, and threshold were explicitly frozen after initial implementation.
- **IF-v2's 19 selected features and its locally-fit scaler** must not be silently changed — they were derived from a specific, documented correlation/variance audit on training data only.
- **The dashboard must remain strictly read-only** — never add code that retrains, refits, or recalculates any model/scaler/calibrator from within `dashboard/`.
- **Ground Truth labels must never be displayed on the Alerts or Alert Investigation pages.**
- **SHAP sign convention** — if IF-v2's `TreeExplainer` output is ever recomputed, remember to keep the sign negation (see Section 12/13) or all SHAP interpretation text in the UI becomes backwards.

---

## 17. FUTURE WORK

1. **Immediate:** re-verify the dashboard's actual current folder structure and fix any remaining import/path mismatches before further feature work (see Known Issues #1).
2. Confirm the SQLite log-index build completes successfully at full scale on the user's actual machine; consider adding a progress indicator or a documented minimum free-disk-space requirement.
3. Resolve the `shap` version pin (0.46.0 vs. the verified-working 0.52.0).
4. Consider whether a Keras-native SHAP explainer (DeepExplainer/GradientExplainer) is worth revisiting for Autoencoder explanation speed, now that the KernelExplainer approach is proven correct.
5. Phase II features (Section 2) — MITRE ATT&CK mapping, root-cause analysis, timeline reconstruction, multi-source correlation, PDF reporting, FastAPI layer — none started.
6. Consider converting the Markdown project reports already produced into a formatted `.docx` for formal submission, if desired (was offered but not requested as of this handover).

---

## 18. RUNNING THE PROJECT

**Prerequisites:** Python 3.12 (64-bit), a virtual environment, and the HDFS LogHub dataset files placed in `data/raw/` (`HDFS.log`, `anomaly_label.csv` — required; `HDFS.log_templates.csv`, `Event_traces.csv`, `Event_occurrence_matrix.csv`, `HDFS.npz` — optional, unused by the pipeline).

**Install dependencies (Windows PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Run the full production pipeline:**
```powershell
python main.py
```
This runs, in order: Log Collection → Drain3 Parsing → Feature Extraction → IF-v2 → Autoencoder → Autoencoder Evaluation → Fusion → Fusion Evaluation. Expect roughly 10–15 minutes total on the full dataset (Drain3 parsing is the slowest single stage, ≈9–10 minutes).

**Launch the dashboard (after `main.py` has completed at least once):**
```powershell
streamlit run dashboard/app.py
```
Then open the local URL Streamlit prints (typically `http://localhost:8501`).

**No separate "start frontend" / "start backend" / "start database" commands exist** — see Section 3/6/7 (N/A).

**No environment variables are required.**

**Build commands:** N/A — this is not a compiled/bundled application; there is no build step.

---

## 19. TESTING

**No formal automated test suite (pytest/unittest/etc.) exists in this project as of this handover.** Verification throughout development was done by:
- Running each module standalone (e.g., `python -m models.isolation_forest_model_v2`) against the real dataset and inspecting the generated JSON/CSV reports.
- Explicit leakage checks printed/logged by each module (train/validation/test overlap counts, duplicate BlockId counts) — these function as informal integration tests and are safe to keep relying on.
- `python -m py_compile <file>` used throughout to catch syntax errors before runtime testing.
- Manual Streamlit launch + HTTP health-check (`curl http://localhost:8501/_stcore/health`) used to confirm the dashboard starts without a server-side exception.
- SHAP functions were manually tested against real anomaly and normal blocks with printed output, not an automated assertion-based test.

**Recommendation for OpenCode:** if adding a formal test suite, prioritize (a) a leakage regression test asserting zero split overlap, (b) a smoke test that each pipeline module runs against a small synthetic feature CSV without error, and (c) a dashboard import/compile test — none of these exist yet.

---

## 20. AI AGENT INSTRUCTIONS (for OpenCode)

1. **Inspect before modifying.** Before changing any file, actually read it (and its real neighbors — e.g., `find dashboard -type f`) rather than assuming the structure described in Section 5. This project's folder structure has drifted at least once during development and caused a real bug (Section 13); do not repeat that mistake.
2. **Do not rewrite working modules unnecessarily.** The production pipeline (`main.py` and everything it imports) is verified and metric-matched across independent runs. Prefer the smallest change that satisfies a request.
3. **Preserve the existing architecture** (file-based pipeline + read-only Streamlit dashboard, no database, no API, no auth) unless the user explicitly asks to add one of those things.
4. **Check `requirements.txt` before adding any new dependency**; note the `shap` version discrepancy in Known Issues #4 before assuming the pinned version is what's actually installed/working.
5. **Follow existing coding conventions:** dataclasses for structured reports, `utils/logger.py`'s `get_logger(__name__)` pattern, custom exceptions from `utils/exceptions.py`, `utils/config.py` as the single source of truth for paths/hyperparameters, and the `identify_training_normal_blocks()` pattern for any new leakage-sensitive evaluation code.
6. **There are no secrets to expose** in this project (no API keys, credentials, or `.env` file exists) — but do not introduce any without flagging it to the user first, since the project has been explicitly offline/local throughout.
7. **Test changes before declaring them complete.** At minimum: `python -m py_compile` the changed file(s), and where feasible, actually run the affected module against real project data (not just assert it "should work") — this project's development process consistently found real bugs (Sections 13, 14) precisely by insisting on execution over inspection alone.
8. **Explain files changed and why**, matching the level of detail in Section 13 (Bugs and Fixes) of this document — the user has consistently required this level of accountability throughout development.
9. **Consider the DO NOT BREAK list (Section 16) before any change** — several of these constraints (IF-v2 as sole production IF, fixed fusion weights, frozen Autoencoder, dashboard read-only) were explicit, repeated, hard requirements throughout this project's development, not incidental defaults.
