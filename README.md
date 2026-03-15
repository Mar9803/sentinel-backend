# 🛡️ SentinelGraph AI - Fraud Detection Engine

**SentinelGraph** is an advanced analytical engine that merges **Graph Theory** and **Machine Learning** to identify anomalies within massive financial datasets.

It utilizes a hybrid pipeline: extracting structural features via **NetworkX** (PageRank, Clustering) and processing them through an **Isolation Forest** model for robust outlier detection.

---

### 1. Prerequisites
* Python 3.11+
* Virtualenv

### 2. Installation
```bash
git clone [https://github.com/Mar9803/sentinel-backend.git](https://github.com/Mar9803/sentinel-backend.git)
cd sentinel-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execution
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. Access the Swagger UI at `/docs` to test the endpoints interactively.

---

## 🏗️ Core Architecture
The system ingests transactions in CSV format, constructs a similarity graph, and applies centrality algorithms before performing ML inference.

```plaintext
┌─────────────┐       ┌─────────────────┐      ┌──────────────┐
│  CSV Ingest │ ───▶ │  Graph metrics  │  ───▶│  ML Outlier │ 
└─────────────┘       └─────────────────┘      └──────────────┘
```

---

## 📄 API at a Glance

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/analyze` | `POST` | Analyzes a CSV file and returns the top 10 suspicious records. |
| `/api/stats` | `GET` | Returns global engine statistics and metrics. |
| `/` | `GET` | System health check (Status: Online). |

---


## 📚 Full Documentation
For in-depth details regarding the architecture, class logic, and customization, please refer to the [**DOCUMENTATION.md**](./DOCUMENTATION.md).

---

## 🛠️ Tech Stack
* **Backend**: `FastAPI`
* **Analisi**: `NetworkX`, `Scikit-Learn`
* **Data**: `Pandas`, `NumPy`
* **Frontend**: `Astro` (Decoupled)