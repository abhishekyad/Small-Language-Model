import psycopg2
import numpy as np

from transformers import pipeline
import nltk
from nltk.corpus import wordnet
from embeddings import embedding_model  # Ensure this imports a valid embedding model

# PostgreSQL connection details
DB_NAME = "rag_db"
DB_USER = "postgres"
DB_PASSWORD = "qwerty"
DB_HOST = "localhost"
DB_PORT = "5432"

# Connect to PostgreSQL
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cur = conn.cursor()
paraphrase_pipeline = pipeline("text2text-generation", model="t5-small")

def expand__with_lsca(question):
    nltk.download("wordnet")
    words = question.split()
    expanded_terms = []

    for word in words:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        expanded_terms.extend(synonyms)

    # Paraphrase the question using a small model (alternative: use OpenAI API)
    paraphrased_question = paraphrase_pipeline(question, max_length=50, num_return_sequences=1)[0]["generated_text"]

    # Combine original, synonyms, and paraphrased question
    expanded_query = list(set(words + expanded_terms))  # Remove duplicates
    return " ".join(expanded_query) + " " + paraphrased_question

def store_document(document_text):
    """Stores a document and its embedding in PostgreSQL."""
    try:
        embedding = embedding_model.embed_query(document_text)  # Generate embedding
        
        # Convert embedding to PostgreSQL array format
        embedding_array = np.array(embedding).tolist()

        # Insert into database
        cur.execute("""
            INSERT INTO documents (content, embedding) 
            VALUES (%s, %s);
        """, (document_text, embedding_array))
        
        conn.commit()
        print("✅ Document stored successfully.")
    
    except Exception as e:
        print(f"❌ Error storing document: {e}")
        conn.rollback()  # Rollback in case of error
    # try:
    #     embedding = embedding_model.embed_query(expand__with_lsca(document_text))  # Generate embedding
        
    #     # Convert embedding to PostgreSQL array format
    #     embedding_array = np.array(embedding).tolist()

    #     # Insert into database
    #     cur.execute("""
    #         INSERT INTO documents (content, embedding) 
    #         VALUES (%s, %s);
    #     """, (document_text, embedding_array))
        
    #     conn.commit()
    #     print("✅ Document stored successfully.")
    
    # except Exception as e:
    #     print(f"❌ Error storing document: {e}")
    #     conn.rollback()  # Rollback in case of error


def query_database(question, top_k=3):
    """Retrieves the top-k most relevant documents based on semantic similarity."""
    try:
        query_embedding = embedding_model.embed_query(question)

        # Ensure the embedding size is 384
        if len(query_embedding) != 384:
            raise ValueError(f"Embedding size mismatch: Expected 384, got {len(query_embedding)}")

        query_embedding = np.array(query_embedding, dtype=np.float32)  # Convert to NumPy array

        cur.execute("""
            SELECT content FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (query_embedding.tolist(), top_k))  # Convert to list before passing

        results = cur.fetchall()
        return [row[0] for row in results]
    
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return []

def close_connection():
    """Closes the database connection."""
    cur.close()
    conn.close()

# Example Usage
if __name__ == "__main__":
    with open("arcadia_dataset.txt", "r", encoding="utf-8") as file:
        content = file.read()

    store_document(content)
