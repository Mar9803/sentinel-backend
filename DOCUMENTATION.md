# 🛡️ SentinelGraph AI - Fraud Detection Engine

**SentinelGraph** is an analytical engine that combines **Graph Theory** and **Machine Learning** to identify anomalies in massive financial datasets. It utilizes a hybrid pipeline: extracting structural features via NetworkX (PageRank, Clustering) and processing them through an Isolation Forest model for robust outlier detection.

---

## 📖 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Key Components](#-key-components)
3. [Tech Stack](#-tech-stack)
4. [Core Pipeline](#-core-pipeline)
5. [Quick Start](#-quick-start)
6. [API Reference](#-api-reference)
7. [Data Sources](#-data-sources)


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
│          └────────▶│ Graph Analysis Engine│       │        │
│                     │ (PageRank/Clustering)│       │        │
│                     └──────────┬───────────┘       │        │
│                                │                   │        │
│                     ┌──────────▼───────────┐       │        │
│                     │  Isolation Forest ML │ ──────┘        │
│                     │  (Anomaly Scoring)   │                │
│                     └──────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

```
---

## ⚙️ Key Components

#### 1. **Detection Pipeline** (`src/`)

**GraphAnalysisEngine** (`graph_analysis.py`):
- Uses `NetworkX` to construct a similarity network of transactions.
-  Extracts structural features: **PageRank** (node importance) and **Clustering Coefficient** (local density).

**IsolationForestModel** (`model.py`):
- Uses `Scikit-Learn` implementation of the **Isolation Forest** algorithm.
- Performs unsupervised anomaly detection by isolating outliers in the high-dimensional feature space.

**TransactionRanker** (Core Logic):
- **Input**: Raw Transaction Data (`CSV`).
- **Output**: Flagged Anomalies with `anomaly_score`.
- **Process**:
  1. **Data Cleaning**: `engineer.clean_data(raw_data)` handles missing values.
  2. **Graph Analysis**: `graph_engine.build_similarity_graph` creates the network.
  3. **Feature Merge**: Combines original financial features with PageRank/Clustering metrics.
  4. **ML Prediction**: `detector.predict(X)` identifies outliers (labeled as `-1`).


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

## 🛠️ Tech Stack

* **Backend**: `FastAPI` (Python 3.11+)
* **Graph Theory**: `NetworkX`
* **Machine Learning**: `Scikit-Learn` (Isolation Forest)
* **Data Handling**: `Pandas`, `NumPy`
* **Frontend**: `Astro` (Separate Repository)

---

## 📈 Core Pipeline


1. **Ingest**: Accepts `CSV` files via `multipart/form-data` endpoints.
2. **Graph Analysis**: Constructs a similarity graph between transactions to detect hidden relationships.
   - **PageRank**: Measures the relative importance and centrality of a transaction within the payment network.
   - **Clustering Coefficient**: Identifies local density of nodes, often signaling organized fraud rings.
3. **ML Inference**: Processes the enriched feature vector (Original data + Graph metrics) using the `sentinel_v1.pkl` model to assign a final `anomaly_score`.

---

## 🚀 Quick Start

#### Clone & Navigate
```bash
git clone [https://github.com/Mar9803/sentinel-backend.git](https://github.com/Mar9803/sentinel-backend.git)
cd sentinel-backend
```
#### Environment Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
source venv/Scripts/activate  
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
#### Launch API

```Bash
uvicorn main:app --reload
API live at http://127.0.0.1:8000. Interactive Swagger UI at /docs.
```




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
```

### Behavior

* **Model Validation**: Checks if the ML Model is loaded and trained (`detector.is_trained`).
* **Data Ingestion**: Reads the file stream into a `Pandas` DataFrame.
* **Graph Extraction**: Computes Graph Centrality metrics (PageRank, Clustering).
* **Result Filtering**: Returns the top 10 anomalies with their statistical scores to minimize network overhead.

---

📊 System Stats
GET /api/stats

Response (200 OK):
```json
{
  "total_processed": 500000,
  "flagged_suspicious": 1240,
  "accuracy_estimate": 0.98,
  "system_health": "healthy"
}
```
---


## 🛠️ Data Sources
**Customizing Data Ingestion**
You can adapt the pipeline to different financial formats by modifying the `engineer` class:

```Python
# Example of custom data cleaning in src/features.py
def clean_data(self, raw_data):
    # Handle specific bank formats (e.g., dropping NaNs)
    df = raw_data.dropna()
    return df
```
**Extending Graph Features**

If you need to add more network metrics (like Betweenness Centrality) to the analysis:

```Python
# src/graph_analysis.py
def extract_new_metrics(self):
    # Using NetworkX to add advanced centrality metrics
    return nx.betweenness_centrality(self.G)
```
