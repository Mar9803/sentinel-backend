# 🛡️ SentinelGraph AI - Fraud Detection Engine

**SentinelGraph** is an analytical engine that combines **Graph Theory** and **Machine Learning** to identify anomalies in massive financial datasets. It utilizes a hybrid pipeline: extracting structural features via NetworkX (PageRank, Clustering) and processing them through an Isolation Forest model for robust outlier detection.

---

## 📖 Table of Contents
1. [Project Overview](#-sentinelgraph-ai---fraud-detection-engine)
2. [Architecture](#️-system-architecture)
3. [Core Pipeline](#-core-pipeline)
4. [API Reference](#-api-reference)
5. [Quick Start](#-quick-start)

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   SentinelGraph API (FastAPI)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐      ┌─────────────┐      ┌──────────┐   │
│   │ POST /analyze│ ────▶│ CSV Ingest  │ ────▶│ Response │   │
│   └──────────────┘      └─────────────┘      └──────────┘   │
│          │                     │                   ▲        │
│          │          ┌──────────▼───────────┐       │        │
│          │          │   Feature Engineer   │       │        │
│          │          │ (Cleaning & Scaling) │       │        │
│          │          └──────────┬───────────┘       │        │
│          │                     │                   │        │
│          │          ┌──────────▼───────────┐       │        │
│          └─────────▶│ Graph Analysis Engine│       │        │
│                     │ (PageRank/Clustering)│       │        │
│                     └──────────┬───────────┘       │        │
│                                │                   │        │
│                     ┌──────────▼───────────┐       │        │
│                     │  Isolation Forest ML │ ──────┘        │
│                     │  (Anomaly Scoring)   │                │
│                     └──────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

---

## ⚙️ Key Components

#### 1. **Detection Pipeline** (`src/`)

**GraphAnalysisEngine** (`graph_analysis.py`):
- Uses `NetworkX` to construct a similarity network of transactions.
- Extracts structural features: **PageRank** (node importance) and **Clustering Coefficient** (local density).

**IsolationForestModel** (`model.py`):
- Uses `Scikit-Learn` implementation of the **Isolation Forest** algorithm.
- Performs unsupervised anomaly detection by isolating outliers in the high-dimensional feature space.

**TransactionRanker** (Core Logic):
- **Input**: Raw Transaction Data (`CSV`).
- **Output**: Flagged Anomalies with `anomaly_score`.
- **Process**:
  1. Data Cleaning & Feature Scaling.
  2. Structural Feature Extraction (Graph Metrics).
  3. ML Inference (Isolation Forest).
  4. Final Ranking based on Anomaly Probability.


#### 2. **Storage & Data Handling** (`models/`)

**In-Memory Model Store**:
- Loads the pre-trained `.pkl` model at startup for near-zero latency inference.
- Handles massive dataframes using `Pandas` and `NumPy` for efficient vectorized operations.

#### 3. **API Layer** (`main.py`)

**POST `/analyze`**:
- Accepts `multipart/form-data` (CSV).
- Orchestrates the full pipeline (Graph + ML).
- Returns detailed results including graph metrics and anomaly scores.

**GET `/api/stats`**:
- Returns cumulative metrics: `total_processed`, `flagged_suspicious`, and `system_health`.

#### 4. **Frontend Dashboard** (`astro-frontend/`)

- Built with **Astro** for extreme performance.
- Decoupled architecture communicating via REST API.
- Real-time visualization of transaction clusters and flagged outliers.

---



🛠️ Tech Stack
Backend: FastAPI (Python 3.11+)

Graph Theory: NetworkX

Machine Learning: Scikit-Learn (Isolation Forest)

Data Handling: Pandas, NumPy

Frontend: Astro (Separate Repository)

📈 Core Pipeline
Ingest: Accepts CSV files via multipart/form-data endpoints.

Graph Analysis: Constructs a similarity graph between transactions.

PageRank: Measures importance/centrality within the network.

Clustering Coefficient: Identifies local density (fraud rings).

ML Inference: Uses sentinel_v1.pkl to assign an anomaly_score.

🚀 Quick Start
1. Clone & Navigate

Bash
git clone [https://github.com/Mar9803/sentinel-backend.git](https://github.com/Mar9803/sentinel-backend.git)
cd sentinel-backend
2. Environment Setup

Bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Launch API

Bash
uvicorn main:app --reload
API live at http://127.0.0.1:8000. Interactive Swagger UI at /docs.

---

## 📄 API Reference

### Analyze Transactions
`POST` `/analyze`

> **Purpose**: Ingests a CSV batch of financial records for real-time anomaly detection.

**Request Body**:
* `file`: `multipart/form-data` (Standard CSV file)

**Response** (`200 OK`):
```json
{
  "total_analyzed": 284807,
  "anomalies_found": 24058,
  "results": [
    {
      "pagerank": 0.000005,
      "clustering": 0.9942,
      "anomaly_score": -0.01
    }
  ]
}
Behavior:

✅ Validates CSV structure and cleans missing data.

✅ Extracts structural features using PageRank and Clustering.

✅ Performs unsupervised inference via Isolation Forest.

📊 System Stats
GET /api/stats

Response (200 OK):

JSON
{
  "total_processed": 500000,
  "flagged_suspicious": 1240,
  "accuracy_estimate": 0.98,
  "system_health": "healthy"
}
📝 Design Decisions
Decoupled Architecture: Backend (Python) and Frontend (Astro) are separated for maximum performance.

In-Memory Loading: The .pkl model is pre-loaded to minimize inference latency.

Security-First: Provides statistical confidence for SOC analyst prioritization.