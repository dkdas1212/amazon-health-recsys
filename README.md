# 🛍️ Amazon Health & Personal Care Recommendation Engine

An end-to-end ML recommendation system built on 281K+ Amazon reviews with a full data engineering pipeline.

## 🏗️ Architecture

```
Amazon Reviews Dataset
        ↓
Apache Kafka (simulated clickstream)
        ↓
Apache Airflow (pipeline orchestration)
        ↓
PySpark (distributed feature engineering)
        ↓
PostgreSQL + dbt (data modeling)
        ↓
SVD + TF-IDF + Hybrid ML Models (RMSE: 0.73)
        ↓
FastAPI (REST API) + caching
        ↓
Streamlit Dashboard (interactive UI)
        ↓
LangGraph Agentic Layer (Groq LLM) — natural-language interface
```

## 📊 Dataset

- **Source:** Amazon Health & Personal Care Reviews
- **Size:** 281,642 reviews (cleaned from 567K raw)
- **Users:** 46,523 unique users
- **Products:** 15,283 unique products
- **Sparsity:** 99.96%

## 🤖 ML Models

| Model | Metric | Score |
|---|---|---|
| SVD Collaborative Filtering | RMSE | 0.73 |
| TF-IDF Content-Based | Avg Cosine Similarity | 0.73 |
| Hybrid Model | Blended Score | 70% CF + 30% CB |

## 🧠 Agentic AI Layer

A LangGraph agent (Groq-backed) sits on top of the FastAPI service, giving
it a natural-language interface instead of raw API calls.

- **Flow:** `parse_intent → select_tool → search_category → apply_constraints → format_response`
- **Real full-catalog search** — a dedicated `/search` endpoint scans all
  reviews for category keywords (coffee, snacks, drinks, pet items, etc.),
  rather than just filtering an already-narrow top-N recommendation list
- **Grounded, hallucination-resistant responses** — the LLM is constrained
  to only reference real `product_id`s and fields actually returned by the
  tools, and is explicitly prompted to admit when data (price, allergens,
  meal/recipe info) isn't available rather than fabricate an answer
- **Explainability logging** — every query, response, and reasoning trace
  is logged to a Postgres `agent_logs` table
- **Chat UI** — Streamlit front end (`agentic_layer/streamlit_chat.py`)

See [`agentic_layer/README.md`](agentic_layer/README.md) for full
architecture, setup, supported query types, and known data limitations.

## 🔎 Semantic Search + Review Q&A (RAG)

Embeds a sample of reviews (MiniLM) into pgvector. `GET /rag/search` does semantic
product search; `POST /rag/ask` answers questions from a product's reviews with
numbered citations and a "not enough evidence" fallback (LLM: Groq `gpt-oss-120b`).
Code lives in `rag/`.

**Evaluation finding:** the review data contains copy-pasted reviews shared across product
variants (58% of reviews in a first 4,000-user sample had duplicate text). On that sample a
content-based eval scored TF-IDF hit@10 = 0.70, which reflects sibling variants, not taste.
After dropping reviews whose text appears on multiple products and subtracting each user's
own reviews, a fresh sample (820 users evaluated, 4,304 products) gave hit@10: popularity
0.077, TF-IDF 0.094, embeddings 0.049. The gap between TF-IDF and popularity is within noise
at this sample size, and embeddings did not beat popularity, so the hybrid keeps CF + TF-IDF.
The RAG features are judged by inspecting cited answers, not by hit@10.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Streaming | Apache Kafka |
| Orchestration | Apache Airflow |
| Processing | Apache PySpark |
| Transformation | dbt |
| Storage | PostgreSQL, pgvector |
| ML | scikit-surprise, scikit-learn |
| Embeddings | sentence-transformers (MiniLM) |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Agentic Layer | LangGraph, Groq API |
| Dashboard | Streamlit |
| Containerization | Docker |

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.9+
- Java 17

### Run the project

```bash
# Clone the repo
git clone https://github.com/dkdas1212/amazon-health-recsys.git
cd amazon-health-recsys

# Start all services
docker-compose up

# Activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start FastAPI
uvicorn api.main:app --reload

# Start Streamlit
streamlit run dashboard/app.py

# (Optional) Start the agentic chat layer
cd agentic_layer
pip install -r requirements.txt
export GROQ_API_KEY=your_key
streamlit run streamlit_chat.py

# (Optional) Start the RAG search / review Q&A API
cd rag
pip install -r requirements-rag.txt
cp .env.example .env        # set RAG_DATABASE_URL and LLM_API_KEY
python ingest_reviews.py    # sample, dedupe, embed reviews into pgvector
python eval_content.py      # popularity vs TF-IDF vs embeddings
uvicorn serve:app --port 8001
```

### Services
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| Agentic Chat | http://localhost:8501 (separate process) |
| RAG API | http://localhost:8001 |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5002 |

## 📡 API Endpoints

```bash
# Get recommendations for a user
GET /recommend?user_id=A3OXHLG6DIBRW8&n=10

# Find similar products
GET /similar/B002QWHJOU?n=10

# Get popular products
GET /popular?n=10

# Get user review history
GET /user/A3OXHLG6DIBRW8/history

# Full-catalog keyword search (used by the agentic layer)
GET /search?keywords=coffee,tea&n=10

# System stats
GET /stats

# RAG API (port 8001)
GET  /rag/search?q=low sugar snack for kids
POST /rag/ask   {"question": "...", "product_id": "B006MONQMC"}
```

## 📁 Project Structure

```
amazon-health-recsys/
├── kafka/
│   ├── producer.py          # Simulates user clickstream
│   └── consumer.py          # Writes events to PostgreSQL
├── airflow/
│   └── dags/
│       ├── daily_ingestion.py
│       ├── weekly_retraining.py
│       └── hourly_features.py
├── ml/
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   ├── hybrid.py
│   ├── pyspark_features.py
│   └── train_with_mlflow.py
├── dbt/
│   └── models/
│       ├── staging/
│       │   ├── stg_reviews.sql
│       │   └── stg_events.sql
│       └── marts/
│           ├── user_interactions.sql
│           └── product_features.sql
├── api/
│   └── main.py              # FastAPI application
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── agentic_layer/
│   ├── agent.py              # LangGraph agent (Groq LLM)
│   ├── tools.py               # Tool wrappers over the FastAPI endpoints
│   ├── db_logger.py           # Postgres explainability logging
│   ├── streamlit_chat.py      # Chat UI
│   └── README.md              # Agentic layer docs
├── rag/
│   ├── ingest_reviews.py      # sample, dedupe, embed into pgvector
│   ├── eval_content.py        # popularity vs TF-IDF vs embeddings (hit@10)
│   ├── api.py                 # /rag/search and /rag/ask router
│   └── serve.py               # standalone server for the RAG API
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 🎯 Key Results
- Improved SVD RMSE from **0.86 → 0.73** through hyperparameter tuning
- Kafka producer/consumer simulating a user clickstream into PostgreSQL (demonstration only; the dataset itself is static)
- Matrix sparsity of **99.96%** handled via SVD factorization
- **3 Airflow DAGs** automating daily ingestion, hourly features, weekly retraining
- **4 dbt models** transforming raw data into ML-ready features
- **PySpark** processing 281K reviews across 46K users and 15K products
- Added a **LangGraph + Groq agentic layer** enabling natural-language
  querying over the recommender, with full-catalog search and
  Postgres-logged reasoning traces for explainability
- Added **pgvector semantic search + RAG review Q&A** with citations and a low-evidence
  fallback, plus a leak-free eval that exposed duplicate-review inflation (see RAG section)

## 👤 Author
**Dibya Kanti Das**
- GitHub: [@dkdas1212](https://github.com/dkdas1212)