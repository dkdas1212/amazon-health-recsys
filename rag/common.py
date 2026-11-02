import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
_model = None

def embed(texts, batch_size=64, progress=False):
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=progress)

def connect(vector=True):
    conn = psycopg2.connect(os.environ["RAG_DATABASE_URL"])
    if vector:
        register_vector(conn)
    return conn
