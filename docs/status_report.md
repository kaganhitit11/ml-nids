# ML-NIDS Project Narrative + Status Report

Date: 2026-01-20

This document is a **single, end-to-end narrative** of what this repository is trying to do (from the proposal), what is currently implemented in code, what outputs it produces, and what gaps remain. It is written so you can reuse it as a backbone for the final report.

---

## 1) One-paragraph summary (what the project is)

This project studies **training-time structured label poisoning** against Machine Learning based **Network Intrusion Detection Systems (NIDS)**. The core idea is: keep features unchanged, but **flip a bounded fraction of training labels** in structured ways (not just random noise), then measure how much models degrade when tested on clean data. The work is **cross-dataset** (several NIDS datasets) and **cross-model-family** (sklearn baselines + PyTorch neural models). It also includes **defenses** that attempt to identify and remove/downweight suspicious training points using model-derived signals (primarily loss).

---

## 2) Proposal intent (ground truth target scope)

### Research question
How robust are common NIDS ML models to **structured label flipping attacks** under a fixed label-flip budget, and how does that robustness generalize **across datasets** and **across model families**?

### Target components described in the proposal
- **Datasets (proposal):**
  - UNSW-NB15
  - LYCOS-IDS2017 (a corrected CIC-IDS2017 variant)
  - CUPID
  - CIDDS-001
  - (The text also says “five benchmarks”, but the table lists 4; so even the proposal has a small counting inconsistency.)
- **Models:** Logistic Regression (LR), Random Forest (RF), MLP, 1D-CNN, RNN
- **Poisoning strategies (proposal):** class-hiding, feature-targeted, influence/loss-aware, disagreement-based, temporal-window
- **Defenses (proposal):** filtering/removal and reweighting based on signals like loss/confidence/disagreement
- **Evaluation outputs:** accuracy, macro/per-class PR/F1, confusion matrices, performance vs poisoning-rate curves

---

## 3) Repository architecture (what exists today)

### Top-level scripts
- `train.py`: the **main training + evaluation** entry point.
- `poisoning.py`: generates **poisoned variants** of the training set under different strategies.
- `dataloaders.py`: dataset-specific PyTorch `Dataset` wrappers + `DataLoader` builders.
- `models.py`: PyTorch models (MLP/RNN/CNN) + sklearn model constructors.
- `process_cic.py`: converts CIC labels to binary in-place.
- `process_cidds.py`: converts CIDDS `attack_type` to binary `label` in-place.

### Batch (HPC) scripts
- `batch_scripts/train.sbatch`: runs clean + poisoned training loops for a dataset.
- `batch_scripts/poison.sbatch`: runs poisoning generation for multiple strategies.

### Data folder in this repo
There is a `data/` folder containing smallish train/test CSVs:
- `data/cic/{train.csv,test.csv,metadata.json}`
- `data/cidds/{train.csv,test.csv}`
- `data/cupid/{train.csv,test.csv}`
- `data/nusw/{train.csv,test.csv,metadata.json}`

**Important:** the Python code generally references `data_real/...` paths, but this repository does not contain a `data_real/` directory right now (so out-of-the-box, many default paths are inconsistent).

---

## 4) What the code actually does (pipeline narrative)

### 4.1 Data loading & preprocessing (`dataloaders.py`)
Each dataset gets its own `torch.utils.data.Dataset` wrapper.

Common pattern:
- Read CSV into pandas.
- Split features vs label.
- Encode categorical columns where needed.
- Standardize numeric columns with `StandardScaler`.
- Return `(features_tensor, label_tensor)` per sample.

Dataset-specific details:

**UNSW-NB15**
- Label column: `label` already binary in the CSV.
- Drops: `id`, `label`, `attack_cat`.
- Categorical columns: `proto`, `service`, `state` encoded with `LabelEncoder`.
- Scaling: only numeric columns.

**CIC-IDS2017**
- Assumes the CSV has numeric `label` where 0=benign and nonzero=attack. It re-binarizes via `(label != 0)`.
- Uses all other columns as numeric features.
- Scaling: all features.

**CUPID**
- Strips column names.
- Finds label column `Label` or `label`.
- Drops network identity + timestamp columns (`Source IP`, ports, `Timestamp`, etc.) and label.
- Scaling: all remaining features.

**CIDDS**
- Assumes a numeric `label` column already exists.
- Drops `original_label`, `label`, `attack_id`.
- Categorical: only `proto` via `LabelEncoder`.
- Scaling: numeric columns.

Loader path convention:
- Each `get_*_dataloaders(data_dir, ...)` reads training from `<data_dir>/train.csv`.
- **BUT** test paths are hard-coded to `data_real/<dataset>/test.csv` rather than `<data_dir>/test.csv`.

This design implies: training can come from poisoned folders, but test is always from the “clean canonical” dataset folder.


### 4.2 Models (`models.py`)
Model suite matches the proposal:
- **Sklearn:**
  - Logistic Regression: `LogisticRegression(max_iter=1000, random_state=42)`
  - Random Forest: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`
- **PyTorch:**
  - `SimpleMLP`: 1 hidden layer + dropout.
  - `SimpleRNN`: reshapes flat features into a sequence (auto-selects a divisor `seq_len`) then runs a basic `nn.RNN`.
  - `SimpleCNN`: 1D conv over features treated as a length-`input_dim` sequence.


### 4.3 Training, evaluation, and outputs (`train.py`)
`train.py` is the main orchestrator.

Inputs (CLI):
- `--dataset`: `nusw|cic|cupid|cidds`
- `--model`: `logistic|random_forest|mlp|rnn|cnn`
- `--data_dir`: dataset folder (defaults to `data_real/{dataset}/`)
- `--epochs`, `--lr`, etc.
- Optional: `--log_per_sample` to save per-sample predictions and confidences.
- Optional: `--defense_strategy` = `removal|reweighting`.

Training flow:
- Loads dataset via dataloaders.
- Computes `input_dim` from a batch.
- Chooses model type:
  - sklearn for LR/RF: extracts all data from dataloaders into NumPy arrays.
  - PyTorch for MLP/RNN/CNN.
- Trains model.
- Evaluates on test set.
- Saves model:
  - sklearn to `models/<dataset>_<model>.pkl`
  - torch to `models/<dataset>_<model>.pth`
- Writes evaluation JSON under `eval_results/{LR|RF|MLP|1D-CNN|RNN}/...` containing:
  - accuracy
  - confusion matrix
  - full `classification_report` dict (includes macro avg / weighted avg)
  - hyperparameters


### 4.4 Defenses implemented (in `train.py`)
Two defenses exist for both sklearn and PyTorch models:

**Removal defense**
- Compute per-sample loss.
- For each class, remove the top `p%` highest-loss points (class-aware).

**Reweighting defense**
- Compute per-sample loss.
- For each class, downweight top `p%` highest-loss points to weight 0.1.

How loss is computed:
- PyTorch: train a temporary model for 2 epochs and compute per-sample cross-entropy.
- Sklearn: uses 4-fold cross-validation to get out-of-fold predicted probabilities and per-sample cross-entropy loss.

Note: This matches the proposal’s “loss-based filtering/reweighting” direction, but there is no explicit “disagreement-driven defense” implemented yet (only disagreement-driven poisoning).


### 4.5 Poisoning generation (`poisoning.py`)
Poisoning creates multiple training variants from `<data_dir>/train.csv` and writes poisoned training CSVs under:

`<data_dir>/poisoned/<strategy>/<PPP>/train.csv`

where `<PPP>` is `005`, `010`, `020` for 5%,10%,20% by default.

Implemented poisoning strategies:

1) **Class-hiding (`class_hiding`)**
- Select samples with `label == target_class` (default target_class=1) to prioritize.
- Under a global budget `K = floor(N * poison_rate)`, flip labels of `K` samples.

2) **Feature-predicate (`feature_predicate`)**
- Uses a dictionary of per-dataset predicates.
- Creates a candidate pool by union-ing all predicate matches.
- Under global budget `K`, prioritize predicate matches; if not enough, fill with random non-matches.

3) **Confidence-based (`confidence_based`)**
- Requires a per-sample “confidence” log file from a previous training run.
- Picks the `K` lowest-confidence samples (globally) to flip.

4) **Disagreement-based (`disagreement`)**
- Requires two per-sample log files (two different seeds).
- Candidates are samples where predicted labels differ.
- If enough candidates, prioritize those with largest confidence difference.

5) **Temporal-window (`temporal`)**
- Implemented for CUPID only (needs `Timestamp`).
- Buckets samples into windows (default 5 minutes).
- Computes z-score style deviation from window mean/std across selected features.
- Flips top `K` globally highest deviation-score samples.

Label flipping rule used in the implementation:
- The poisoning code uses a **bidirectional flip**: 0→1 and 1→0 for selected indices.
- This is not purely “hide attacks as benign”; it’s symmetric corruption. The printed stats do track how many flips are “attack→benign” vs “benign→attack”.

---

## 5) Proposal vs implementation: alignment tables

### 5.1 Datasets

| Item | Proposal | Implemented in code | Present in repo data | Notes |
|---|---|---|---|---|
| UNSW-NB15 | Yes | Yes (`nusw`) | Yes (`data/nusw/`) | Uses `proto/service/state` categorical encoding |
| LYCOS-IDS2017 | Yes | No | No | Code uses `cic` instead; proposal references corrected CIC (LYCOS) |
| CIC-IDS2017 | Mentioned indirectly | Yes (`cic`) | Yes (`data/cic/`) | `process_cic.py` makes binary labels |
| CUPID | Yes | Yes (`cupid`) | Yes (`data/cupid/`) | Temporal poisoning is built specifically for CUPID |
| CIDDS-001 | Yes | Yes (`cidds`) | Yes (`data/cidds/`) | Needs processing for numeric label |

**Bottom line:** repo currently supports **4 datasets** (nusw, cic, cupid, cidds). LYCOS is not implemented.


### 5.2 Models

| Model family | Proposal | Implemented | Where |
|---|---:|---:|---|
| Logistic Regression | Yes | Yes | `models.py` + `train.py` |
| Random Forest | Yes | Yes | `models.py` + `train.py` |
| MLP | Yes | Yes | `models.py` + `train.py` |
| 1D-CNN | Yes | Yes | `models.py` + `train.py` |
| RNN | Yes | Yes | `models.py` + `train.py` |


### 5.3 Poisoning strategies

| Strategy | Proposal | Implemented | Notes |
|---|---:|---:|---|
| Class-hiding | Yes | Yes | Implementation flips labels bidirectionally under global budget |
| Feature-targeted | Yes | Yes | Predicate definitions exist, but see “schema mismatch” risks below |
| Loss-aware / influence-aware | Yes | Partial | Loss is used as a defense; poisoning-by-loss is not present |
| Disagreement-based | Yes | Yes | Requires per-sample logs from two seeds |
| Temporal-window | Yes | Yes | Implemented for CUPID only |


### 5.4 Defenses

| Defense | Proposal | Implemented | Notes |
|---|---:|---:|---|
| Filtering/removal by suspiciousness | Yes | Yes (loss-based) | Class-aware top p% loss removal |
| Reweighting | Yes | Yes (loss-based) | Downweights top p% loss to 0.1 |
| Disagreement-based defense | Mentioned | No | Only poisoning uses disagreement |
| Composite suspiciousness | Mentioned | No | Not implemented |


### 5.5 Metrics and result storage

| Metric/output | Proposal | Implemented |
|---|---:|---:|
| Accuracy | Yes | Yes |
| Per-class precision/recall/F1 | Yes | Yes (via sklearn `classification_report`) |
| Macro average F1 | Yes | Yes (included in report dict) |
| Confusion matrix | Yes | Yes |
| Poisoning-rate degradation curves | Yes | Not automated | Can be created later by aggregating `eval_results/*.json` |

---

## 6) Current “as-is” status, risks, and inconsistencies (important for your final report narrative)

### 6.1 The biggest operational mismatch: `data_real/` vs `data/`
- Default paths in `train.py` and dataloader test paths assume `data_real/<dataset>/...`.
- This repo contains `data/<dataset>/...`.

Practical impact:
- Running `python train.py --dataset cic --model mlp` will fail unless you create a `data_real/` directory or pass `--data_dir data/cic/`.
- Even when training from `data/cic/`, **testing still tries to read** `data_real/cic/test.csv` because of the hard-coded test path in `dataloaders.py`.

This is the #1 thing to document as “current code expects a different data root”.


### 6.2 Feature predicate poisoning likely doesn’t do what it claims (schema mismatches)
The predicate poisoning relies on column names that must match exactly.

- **CIC:** predicates use names like `Flow Duration`, `Total Fwd Packets`, `Average Packet Size`, `SYN Flag Count`.
  - But the repo’s `data/cic/train.csv` columns are underscore-style: `Flow_Duration`, `Total_Fwd_Packets`, `Average_Packet_Size`, `SYN_Flag_Count`, etc.
  - In `check_predicate`, missing columns produce all-False masks.
  - Result: candidate pool becomes empty and poisoning falls back to random non-matching samples → effectively **random label flips**, not feature-targeted.

- **CIDDS:** predicates check `proto == tcp`.
  - In `data/cidds/train.csv`, proto values look like `TCP  ` with extra spaces.
  - `check_predicate` lowercases strings but does not strip whitespace.
  - Result: predicate matches may again be empty → fallback random flips.

For a final report, you can either:
- explicitly acknowledge this as an “implementation caveat / threat to validity”, or
- fix it and rerun, but as of now, it’s a real mismatch.


### 6.3 CIDDS needs preprocessing before training
Current `data/cidds/train.csv` has `label` values like `normal` (string) and an `attack_type` column.

- `process_cidds.py` converts `attack_type` to numeric `label` and preserves the old label as `original_label`.
- `CIDDS_Dataset` expects numeric `label`.

So, status-wise: CIDDS training is not “ready” unless preprocessing was run somewhere else.


### 6.4 Poisoning logs are hard-coded to an HPC home directory
`poisoning.py` expects confidence/disagreement log files under paths like:
- `/home/ohitit20/<log_dir>/<dataset>_<model>/<dataset>_<model>_seed42_per_sample_metrics.csv`

This is consistent with the `.sbatch` scripts but not portable to local runs.


### 6.5 Test CSV path recorded in results may be wrong
At the end of `train.py`, it records:
- `train_csv = <data_dir>/train.csv` (correct for poisoned training)
- `test_csv = <data_dir>/test.csv`

But in the implemented dataloaders, the test set is often read from `data_real/<dataset>/test.csv` regardless of `data_dir`.
So the stored `test_csv` path in the JSON results may not reflect reality.

---

## 7) Experiment matrix (what you can claim you ran / can run)

Define:
- Datasets $D = 4$ (`nusw`, `cic`, `cupid`, `cidds`)
- Models $M = 5$ (LR, RF, MLP, RNN, CNN)
- Poisoning rates $R = 3$ (5%, 10%, 20%) + optionally clean baseline
- Poisoning strategies $S = 5$ (class_hiding, feature_predicate, confidence_based, disagreement, temporal)
- Defenses $F = 3$ (none, removal, reweighting)

A full factorial would be huge; realistically, the repo is currently most “ready” for:
- Clean baseline (no poisoning) across all models and datasets.
- Poisoning by class_hiding/feature_predicate across all datasets.
- Temporal only for CUPID.
- Confidence/disagreement only if per-sample logs exist.

Suggested “reportable” minimal matrix (high signal, feasible):
- For each dataset and model: clean baseline + class_hiding at 5/10/20.
- Add defenses on top of one or two model families (e.g., LR and MLP) to show robustness trade-offs.

---

## 8) What outputs exist (and what to include in the final report)

When you run `train.py`, you get:
- A trained model file under `models/`.
- One evaluation JSON under `eval_results/` containing:
  - accuracy
  - confusion matrix
  - per-class + macro/weighted metrics
  - hyperparameters (including defense strategy when used)

To build the final report figures:
- Parse all `eval_results/**.json`.
- Group by dataset, model, attack_type, poisoning_percentage, defense.
- Plot accuracy and macro-F1 vs poisoning rate.

---

## 9) “Status report” conclusion (where you are now)

### What is solid and can be described confidently
- The repository implements a complete **train → evaluate → save JSON** loop.
- The model suite matches the proposal (LR/RF/MLP/RNN/CNN).
- The poisoning module implements several structured strategies and produces organized poisoned training sets.
- The defense module implements class-aware removal/reweighting based on per-sample loss.

### What is incomplete or threatens validity
- Dataset root mismatch (`data_real/` expected, `data/` present).
- Feature predicate poisoning likely degenerates to random flipping for at least CIC (and probably CIDDS) due to schema/value mismatch.
- Confidence/disagreement poisoning are not portable due to hard-coded absolute paths.
- LYCOS-IDS2017 is not present; the repo currently evaluates CIC instead.

### Recommended immediate next steps (so the final report narrative is clean)
1) Decide whether the canonical data root should be `data/` or `data_real/` and make it consistent.
2) Ensure CIDDS is processed (numeric label) and document preprocessing.
3) Fix predicate feature names to match the actual CSV schemas (or normalize column names before predicate application).
4) Make confidence/disagreement poisoning log paths configurable (relative paths), then regenerate poisoned datasets.
5) Aggregate `eval_results` into a single results table (CSV) + plots for the final report.

---

## Appendix A: Quick “how to run” (current code assumptions)

Because the repo currently lacks `data_real/`, runs will likely require explicit `--data_dir` and/or a `data_real` folder.

Examples (conceptual):
- Clean training (but note test path issue):
  - `python train.py --dataset cic --model mlp --data_dir data/cic/ --epochs 3`

- Generate poisoned training sets (writes under `data/cic/poisoned/...`):
  - `python poisoning.py --dataset cic --strategy class_hiding --data_dir data/cic/`

---

## Appendix B: Mapping: proposal → code files

- Datasets & preprocessing: `dataloaders.py`, `process_cic.py`, `process_cidds.py`
- Models: `models.py`
- Poisoning: `poisoning.py`
- Defenses + evaluation protocol: `train.py`
- HPC reproducibility entry points: `batch_scripts/*.sbatch`
