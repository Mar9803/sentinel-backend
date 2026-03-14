# 🛡️ SentinelGraph AI - Fraud Detection Engine

**SentinelGraph** is an advanced analytical engine that combines **Graph Theory** and **Machine Learning** to identify anomalies in massive financial datasets. It utilizes a hybrid pipeline: extracting structural features via NetworkX (PageRank, Clustering) and processing them through an Isolation Forest model for robust outlier detection.

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

📄 API Reference
Analyze Transactions
POST /analyze

Purpose: Ingests a CSV batch of financial records for real-time anomaly detection.

Request Body:

file: multipart/form-data (Standard CSV file)

Response (200 OK):

JSON
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