# 🛡️ AI Intrusion Detection System (IDS)

> **Wireshark-style live network traffic classifier powered by Machine Learning**  
> Built on the NSL-KDD dataset — visualize, simulate, and detect cyber attacks in real time.

![Live Dashboard Preview](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [How It Works — End to End](#-how-it-works--end-to-end)
- [Dataset](#-dataset-nsl-kdd)
- [ML Models](#-ml-models)
- [Dashboard Features](#-dashboard-features)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Technical Deep-Dive](#-technical-deep-dive)

---

## 🎯 What This Project Does

This is a **live intrusion detection simulator** that mimics what a real network security operations center (SOC) would look like. It:

1. **Simulates real network traffic** — packets flow in continuously, just like Wireshark
2. **Classifies every packet with ML** — Normal, DoS, Probe, R2L, or U2R
3. **Shows you the attack type, confidence score, and protocol** on hover
4. **Lets you switch between 4 different ML models** to compare their behavior
5. **Fires real-time alerts** whenever a threat is detected

No need to pre-select what attack to inject, the ML models classify it themselves.

---

## 🔄 How It Works — End to End

```
Raw NSL-KDD Data
      │
      ▼
┌─────────────────┐
│  preprocess.py  │  ← Cleans, encodes, scales 41 network features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   train.py      │  ← Trains 4 ML models, saves .pkl files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  simulate.py    │  ← Streams packets from the test dataset
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   predict.py    │  ← Runs selected model + multi-class model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  dashboard.py   │  ← Streamlit UI: graph, feed table, alerts, tabs
└─────────────────┘
```

### Step 1 — Preprocessing (`src/preprocess.py`)

- Loads `KDDTrain+.txt` and `KDDTest+.txt` from the `data/` folder
- Drops `difficulty_level` column (not a network feature)
- Maps 39 specific attack names → 4 categories (see [Dataset](#-dataset-nsl-kdd))
- **Encodes** 3 categorical columns (`protocol_type`, `service`, `flag`) using `LabelEncoder`
- **Scales** all 41 features to `[0, 1]` using `MinMaxScaler`
- Saves `scaler.pkl`, `label_encoders.pkl`, `feature_cols.pkl` to `models/`
- Outputs both binary labels (0=normal, 1=attack) and multi-class labels (0–4)

### Step 2 — Model Training (`src/train.py`)

Trains 4 models on the preprocessed data and saves them as `.pkl` files:

| Model | Type | Details |
|---|---|---|
| `random_forest.pkl` | Binary classifier | 100 trees, max_depth=20, `n_jobs=-1` |
| `random_forest_multi.pkl` | Multi-class classifier | Same config, 5-class output |
| `mlp_model.pkl` | Binary neural net | layers=(128,64), early stopping |
| `isolation_forest.pkl` | Unsupervised anomaly detector | 150 trees, trained on **normal traffic only** |

Also saves `feature_importance.pkl` and `metrics.pkl` for the dashboard analysis tabs.

### Step 3 — Simulation (`src/simulate.py`)

Three stream modes:

| Function | What it does |
|---|---|
| `normal_burst(df, n)` | Samples n rows of purely normal traffic |
| `realistic_mixed_stream(df, n, ratio)` | Mixes all attack types in realistic proportions: DoS 50%, Probe 30%, R2L 15%, U2R 5% |
| `mixed_stream(df, n, ratio, type)` | Legacy: picks a specific attack type |

The `realistic_mixed_stream` function is what powers the **"Simulate Attack Event"** toggle — you don't choose the attack type, the model detects it.

### Step 4 — Prediction (`src/predict.py`)

`predict_single(features_scaled, model_name)` returns:

```python
{
    'label': 'Attack',          # or 'Normal'
    'binary': 1,                # 0 or 1
    'confidence': 0.97,         # model's confidence score
    'attack_type': 'DoS',       # specific category or 'Anomaly'
    'model': 'Random Forest',   # display name
}
```

**Dual-inference logic:** For binary models (RF, MLP, Isolation Forest), the dashboard **always runs the multi-class model in parallel** to get the specific attack label (DoS/Probe/R2L/U2R). The primary model's confidence score is used as-is. This means you always see the attack *type* regardless of which model is selected.

### Step 5 — Dashboard (`dashboard.py`)

A full Streamlit application with live simulation, charts, alerts, and analysis.

---

## 📊 Dataset: NSL-KDD

The [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) dataset is the benchmark for IDS research. It contains **41 network connection features** extracted from TCP/IP traffic.

**Files used:**
- `data/KDDTrain+.txt` — Training set (~125,000 samples)
- `data/KDDTest+.txt` — Test set / simulation source (~22,500 samples)

**Attack categories mapped from 39 specific attack names:**

| Category | What it is | Example attacks |
|---|---|---|
| **Normal** | Legitimate traffic | — |
| **DoS** | Denial of Service — floods/overwhelms target | `neptune`, `smurf`, `teardrop`, `back` |
| **Probe** | Reconnaissance — scanning ports/services | `ipsweep`, `nmap`, `portsweep`, `satan` |
| **R2L** | Remote-to-Local — unauthorized remote access | `guess_passwd`, `ftp_write`, `phf`, `imap` |
| **U2R** | User-to-Root — privilege escalation | `buffer_overflow`, `rootkit`, `loadmodule` |

**Key features used by the models:**
`src_bytes`, `dst_bytes`, `duration`, `protocol_type`, `service`, `flag`, `count`, `srv_count`, `serror_rate`, and 32 more network statistics.

---

## 🤖 ML Models

### 🌳 Random Forest (Binary) — `random_forest.pkl`
- **Task:** Normal vs Attack (binary)
- **How it works:** Ensemble of 100 decision trees, each voting on the classification. Feature importances reveal which network features matter most.
- **Confidence:** Probability of the winning class from `predict_proba()`
- **Best for:** High accuracy on known attack patterns

### 🌳 Random Forest (Multi-class) — `random_forest_multi.pkl`
- **Task:** Normal / DoS / Probe / R2L / U2R (5 classes)
- **How it works:** Same ensemble architecture but trained on 5-class labels. Returns class probabilities for all 5 categories.
- **Confidence:** Max probability across 5 classes
- **Best for:** Identifying the specific attack type. Also used as auxiliary model for all other modes.

### 🧠 MLP Neural Network — `mlp_model.pkl`
- **Task:** Normal vs Attack (binary)
- **Architecture:** Input → 128 → 64 → Output, trained up to 300 epochs with early stopping (15% validation split)
- **How it works:** Learns non-linear decision boundaries that tree-based models can miss
- **Confidence:** Probability from `predict_proba()`
- **Best for:** Detecting novel attack patterns that don't fit rigid rules

### 🔍 Isolation Forest — `isolation_forest.pkl`
- **Task:** Anomaly detection (unsupervised)
- **How it works:** Trained **only on normal traffic** — it learns what "normal" looks like, then flags anything that deviates. No attack labels needed during training.
- **Confidence:** Derived from `score_samples()` — more negative = more anomalous
- **Best for:** Zero-day attacks and unknown threats (doesn't need to have seen the attack before)

---

## 🖥️ Dashboard Features

### Live Network Traffic Monitor

The main graph updates with every batch run. Each dot = one packet.

| Visual | Meaning |
|---|---|
| 🟢 Green dots, dashed line | Normal traffic |
| 🔴 Red diamonds | DoS attack |
| 🟠 Orange diamonds | Probe / port scan |
| 🩷 Pink diamonds | R2L (Remote-to-Local) |
| 🟣 Purple diamonds | U2R (User-to-Root) |
| Red shaded zones | Burst of attack packets |

**Hover over any node** → tooltip shows: attack type, confidence %, protocol, service, src/dst bytes, duration, timestamp, model used.

### Sidebar Controls

| Control | What it does |
|---|---|
| **Active ML Model** | Switch between the 4 models instantly — updates classification on next batch |
| **⚡ Simulate Attack Event** | OFF = normal monitoring traffic. ON = injects realistic mix of all 4 attack types |
| **Packets per batch** | 5–60 packets injected per Run Batch click |
| **▶ Run Batch** | Inject one batch, classify each packet, update everything |
| **🗑 Clear** | Reset all packet history, alerts, and charts |
| **🔄 Auto-stream** | Continuously runs batches at set interval (0.5s–5s) — fully autonomous simulation |

### Live KPIs (top row)

| KPI | What it shows |
|---|---|
| Packets Captured | Total packets processed this session |
| Safe Traffic | Count of Normal-classified packets |
| Threats Detected | Count of attack-classified packets |
| Threat Level % | Threats / Total (color-coded: green < 20%, yellow < 40%, red ≥ 40%) |
| Live Accuracy | How often model prediction matches actual label |
| Avg Confidence | Mean confidence score across all packets |

### Live Packet Feed Table

Wireshark-style scrollable table showing the last 35 packets:

```
TIME          PROTO  SERVICE  SRC BYTES  DST BYTES  DUR    ML LABEL  CONF     ACTUAL
13:31:07.986  tcp    private  0          0          0s     [DoS]     100.0%   DoS
13:31:07.165  tcp    other    0          0          0s     [Probe]   98.0%    Probe
13:31:08.108  tcp    http     291        8,625      0s     [Normal]  100.0%   normal
```

- Red-bordered rows = attack packets
- Green-bordered rows = normal traffic
- Color-coded label pills: red for DoS, orange for Probe, pink for R2L, purple for U2R, green for Normal

### Threat Alert Log

Real-time log in the right panel, showing every detected attack:
```
🚨 [13:31:07.986] DoS detected | Conf 100.0% | tcp/private | Random Forest
🚨 [13:31:07.165] Probe detected | Conf 98.0% | tcp/other | Random Forest
```

### Traffic Type Donut Chart

Live breakdown of all traffic types in the current session — updates after every batch.

### Threat Level Gauge

Semi-circular gauge showing current threat percentage (0–100%).

### Analysis Tabs (below simulation)

| Tab | Contents |
|---|---|
| 📈 Feature Importance | Bar chart of top 15 features + table. Shows which network stats matter most for detection |
| 📊 Model Comparison | Accuracy, F1, Precision, Recall bar chart for all 4 models side by side |
| 🧮 Confusion Matrix | RF binary classifier TP/TN/FP/FN heatmap — per-class breakdown table |
| 📋 Session Summary | Attack type breakdown, protocol distribution, top flagged packets table |
| 🧠 How AI Works | Plain-language explanation of traditional IDS vs ML-based IDS, with comparison table |

---

## 📁 Project Structure

```
ai-ids-project/
│
├── dashboard.py              # Main Streamlit application
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Data loading, encoding, scaling
│   ├── train.py              # Model training & evaluation
│   ├── simulate.py           # Traffic stream generators
│   └── predict.py            # Single/batch inference
│
├── data/                     # NSL-KDD dataset (not in repo)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── models/                   # Trained models (not in repo)
│   ├── random_forest.pkl
│   ├── random_forest_multi.pkl
│   ├── mlp_model.pkl
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.pkl
│   ├── feature_importance.pkl
│   └── metrics.pkl
│
├── .streamlit/
│   └── config.toml           # Dark theme configuration
│
├── requirements.txt
└── .gitignore
```

> **Note:** `data/` and `models/` are in `.gitignore` because they contain large files. Follow the setup steps below to generate them locally.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Ananya-Bharti/AI-Intrusion-Detection-System.git
cd AI-Intrusion-Detection-System
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the NSL-KDD dataset

Download from [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)

Place the files in the `data/` folder:
```
data/
  KDDTrain+.txt
  KDDTest+.txt
```

### 5. Preprocess & train models

```bash
# Option A: Train all at once
python src/train.py

# Option B: Step by step
python src/preprocess.py   # creates scaler.pkl, label_encoders.pkl
python src/train.py        # creates all 4 model .pkl files
```

Training takes ~1–3 minutes. You'll see accuracy/F1/confusion matrix for each model.

---

## 🚀 Running the Project

```bash
venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

### Quick walkthrough

1. **Initial view** — Dashboard loads with all KPIs at 0, empty graph
2. **Run normal traffic** — Click **▶ Run Batch** (attack toggle OFF) — normal packets flow in, graph shows green dots
3. **Simulate an attack** — Toggle **⚡ Simulate Attack Event** ON, click **▶ Run Batch**
   - Mixed DoS, Probe, R2L, U2R packets inject automatically
   - Graph shows colored attack markers
   - Feed table shows specific attack types in ML LABEL column
   - Alert log fires for each detected threat
4. **Switch models** — Change the model dropdown mid-session, run another batch — compare how each model scores confidence
5. **Auto-stream** — Toggle **🔄 Auto-stream** ON for continuous live simulation
6. **Analyze** — Scroll down to the analysis tabs for feature importance, model comparison, confusion matrix

---

## 🔬 Technical Deep-Dive

### Dual-Inference Pattern

The key innovation of the dashboard: when you select a binary model (RF Binary, MLP, Isolation Forest), it can only tell you *"attack or not"*. To always show the specific attack type, the dashboard runs **two inferences per packet**:

```python
# 1. Primary model → confidence + binary flag
result = predict_single(features, model_choice)

# 2. Multi-class RF always runs in parallel → attack category
mc_result = predict_single(features, 'random_forest_multi')

# Decision logic
if result['binary'] == 1:
    detected_type = mc_result['attack_type']  # DoS / Probe / R2L / U2R
else:
    detected_type = 'Normal'
```

This gives the best of both worlds — the selected model's confidence, the multi-class model's category.

### Isolation Forest: No Labels Needed

Isolation Forest is trained **exclusively on normal traffic**. It works by randomly partitioning the feature space — anomalies (attacks) are isolated in fewer splits because they're rare and different. The `score_samples()` output is converted to a `[0, 1]` confidence score.

This is the only model that can theoretically detect **zero-day attacks** it has never seen, because it's not looking for known patterns — it's looking for anything that deviates from normal.

### Traffic Simulation Realism

`realistic_mixed_stream()` samples from the actual NSL-KDD test set in proportions that mirror real-world attack distributions:

```
DoS   → 50% of attack traffic  (volumetric, most common in real networks)
Probe → 30% of attack traffic  (scanning/reconnaissance)
R2L   → 15% of attack traffic  (credential/exploitation attacks)
U2R   →  5% of attack traffic  (privilege escalation, rarest)
```

All samples are shuffled before display, so attacks arrive interleaved with normal traffic — just like a real network stream.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | ML models (RF, MLP, IsolationForest), preprocessing |
| `plotly` | Interactive charts (live graph, donut, gauge, heatmap) |
| `joblib` | Model serialization (save/load .pkl files) |
| `matplotlib` / `seaborn` | Static plot support |
| `shap` | (Optional) Model explainability |

---

*Built for VIT — Computer Applications in Information Security (CAIS) Capstone Project*
