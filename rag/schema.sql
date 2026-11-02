CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS review_chunks;
CREATE TABLE review_chunks (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL,
    product_id TEXT NOT NULL,
    rating     INT,
    content    TEXT NOT NULL,
    embedding  vector(384) NOT NULL
);
CREATE INDEX ON review_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON review_chunks (product_id);
