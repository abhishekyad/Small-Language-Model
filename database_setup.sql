CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE docc (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)  -- Adjust to match your embedding model
);