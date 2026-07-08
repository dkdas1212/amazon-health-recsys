# 🛍️ Amazon Health & Personal Care Recommendation Engine

An end-to-end ML recommendation system built on 281K+ Amazon reviews with a full data engineering pipeline.

## 🏗️ Architecture Amazon Reviews Dataset
↓
Apache Kafka (real-time clickstream)
↓
Apache Airflow (pipeline orchestration)
↓
PySpark (distributed feature engineering)
↓
PostgreSQL + dbt (data modeling)
↓
SVD + TF-IDF + Hybrid ML Models (RMSE: 0.73)
↓
FastAPI (REST API) + Redis (caching)
↓
Streamlit Dashboard (interactive UI)## 📊 Dataset
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

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Streaming | Apache Kafka |
| Orchestration | Apache Airflow |
| Processing | Apache PySpark |
| Transformation | dbt |
| Storage | PostgreSQL |
| ML | scikit-surprise, scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI |
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
```

### Services
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
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

# System stats
GET /stats
```
## 📁 Project Structure
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
├── docker-compose.yml
├── Dockerfile
└── requirements.txt

## 🎯 Key Results
- Improved SVD RMSE from **0.86 → 0.73** through hyperparameter tuning
- Real-time clickstream pipeline streaming **10K+ events/day**
- API response time under **100ms** with caching
- Matrix sparsity of **99.96%** handled via SVD factorization

## 👤 Author
**Dibya Kanti Das**
- GitHub: [@dkdas1212](https://github.com/dkdas1212)
