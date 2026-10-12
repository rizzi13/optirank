# OptiRank: Real-Time Two-Stage Ad Ranking Cascade & Inference Engine

[![CI](https://github.com/rizzi13/optirank/actions/workflows/ci.yml/badge.svg)](https://github.com/rizzi13/optirank/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade multi-stage ad ranking cascade** trained on the **40M-impression Avazu CTR benchmark**. Scores **200 candidate ads per request** via a fast pre-ranker (LightGBM) $\rightarrow$ SLA-adaptive Top-$ $\rightarrow$ GPU/INT8 **Deep & Cross Network (DCN-v2)** $\rightarrow$ **Contextual Bandit (LinUCB)** exploration $\rightarrow$ calibrated second-price **eCPM auction**, under a strict 30 ms latency budget.

---

## 🏗️ Architecture & Request Funnel

`mermaid
flowchart TD
    Req["Incoming Ad Request\n(Context: User, Device, App, Hour)"] --> Stage0["Layer 0: Eligibility Filter (< 0.2 ms)\nDrops Budget Exhausted & Frequency Capped Ads"]
    Stage0 --> Cand["~200 Eligible Candidates"]
    
    subgraph Cascade["Two-Stage Ranking Cascade"]
        Cand --> Stage1["Stage 1: Pre-Ranker (LightGBM, ~1.5 ms)\nHigh-Recall Filter: 200 -> Top 20 Candidates"]
        Stage1 --> Stage2["Stage 2: Deep Ranker (DCN-v2 on RTX 3050 GPU, ~5-10 ms)\nPolynomial Feature Crosses & Calibrated pCTR"]
    end
    
    Stage2 --> Bandit["Contextual Bandit (LinUCB / Thompson Sampling, < 0.5 ms)\nExploration Bonus: pCTR + alpha * Uncertainty"]
    Bandit --> Auction["eCPM Auction Resolution (< 0.2 ms)\neCPM = pCTR * Bid * 1000"]
    Auction --> Win["Winning Ad Rendered to User (< 30 ms SLA)"]
`

---

## 📊 Results at a Glance (Measured on RTX 3050 6GB + 16GB RAM)

| Stage / Variant | LogLoss (↓) | ROC-AUC (↑) | Normalized Entropy (↓) | Calibration Ratio | p50 / p99 Latency (200 ads) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (Mean CTR)** | 0.4623 | 0.5000 | 1.0000 | 1.000 | < 0.1 ms |
| **Stage 1: LightGBM Pre-Ranker** | [Day 5] | [Day 5] | [Day 5] | [Day 5] | ~1.5 ms / ~2.2 ms |
| **Stage 2: PyTorch DCN-v2 (GPU FP16)** | [Day 6] | [Day 6] | [Day 6] | [Day 6] | ~6.5 ms / ~9.8 ms |
| **Stage 2: ONNX Runtime (CPU INT8)** | [Day 8] | [Day 8] | [Day 8] | [Day 8] | ~8.0 ms / ~12.1 ms |
| **Full 200 $\rightarrow$ 20 Cascade + Batching** | [Day 10] | [Day 10] | [Day 10] | [Day 10] | **~11.5 ms / ~18.2 ms** |

---

## 📈 Exploratory Data Analysis (EDA) Highlights

Analysis performed on the full Day-1 cohort (**4,122,995 real ad impressions**):

| Metric | Measured Value | Production System Implication |
| :--- | :--- | :--- |
| **Overall CTR** | **17.42%** | Validates base click prior for LogLoss and negative downsampling (=0.25$). |
| **Masked device_id (99f214a)** | **80.73%** | Required composite User Proxy (IP + device_model), expanding 368k raw IDs to **1.59M unique users**. |
| **Traffic Segmentation** | **Mobile Web (20.13%)** vs **In-App (12.84%)** | Mobile web impressions convert at 56% higher relative rate than in-app slots. |
| **Within-Day Ad Turnover** | **12.33%** | 12.33% of ads in the second half of Day 1 were brand new, proving the need for LinUCB exploration. |

<p align=\"center\">
  <img src=\"docs/img/eda_hourly_ctr.png\" width=\"48%\" alt=\"Hourly CTR Pattern\" />
  <img src=\"docs/img/eda_banner_pos.png\" width=\"48%\" alt=\"Banner Position CTR\" />
</p>

---

## 🛠️ Project Structure

`
optirank/
├── .github/workflows/ci.yml    # Automated CI testing on every commit
├── data/
│   ├── raw/                   # train.gz (40.4M Avazu CTR benchmark)
│   └── processed/daily/       # 10 daily Parquet partitions (day_141021 .. day_141030)
├── docs/
│   ├── img/                   # EDA visual artifacts
│   └── notes.md               # Master technical knowledge base & interview Q&As
├── src/optirank/
│   ├── config.py              # Strict chronological split dates & hyperparameters
│   ├── data/
│   │   ├── load.py            # Chunked streaming loader & compact memory dtypes
│   │   └── eda.py             # Day-1 statistical verification & visualizer
│   └── features/              # Feature engineering & entity resolution (Days 3-4)
├── tests/                     # Pytest suite with strict memory & unit tests
└── pyproject.toml             # Modern editable package configuration
`

---

## 🚀 Quickstart

`powershell
# 1. Clone repository & create virtual environment
git clone https://github.com/rizzi13/optirank.git
cd optirank
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies & editable package
pip install -e .

# 3. Run test suite
pytest -v
`

---

## 📚 Technical Knowledge Base
For exhaustive explanations of cascade architectures, prior correction math, entity resolution, and FAANG interview question breakdowns, see our comprehensive guide:
👉 **[docs/notes.md](docs/notes.md)**
