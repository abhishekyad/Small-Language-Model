import numpy as np
import pandas as pd
import ast  # Import ast for safe string to list conversion
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import embedding_model  # Ensure this imports a valid embedding model

def get_embeddings_from_file(filepath="documents_table.txt"):
    """Reads embeddings from a txt file, parses them, and returns a DataFrame."""
    try:
        df = pd.read_csv(filepath, sep='\t')
        embedding_cols = [col for col in df.columns if col.startswith("embedding")]

        for col in embedding_cols:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x))  # Safely parse string to list

        # Convert embedding columns to a NumPy array of floats
        embeddings = np.array(df[embedding_cols].apply(lambda row: np.array(row.tolist())).tolist(), dtype=np.float32)

        return df, embeddings

    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None, None
    except Exception as e:
        print(f"❌ Error reading embeddings from file: {e}")
        return None, None

def query_database(question, top_k=3):
    """Retrieves the top-k most relevant documents based on embeddings from DataFrame."""
    try:
        query_embedding = embedding_model.embed_query(question)
        query_embedding = np.array(query_embedding).reshape(1, -1)  # Reshape for cosine_similarity

        df, embeddings = get_embeddings_from_file()

        if df is None or embeddings is None:
            return []

        similarities = cosine_similarity(query_embedding, embeddings)[0] # Calculate cosine similarities

        top_indices = np.argsort(similarities)[::-1][:top_k] # Find top k indices

        results = []
        contents = df['content'].tolist() # get the content from the dataframe.

        for index in top_indices:
            if index < len(contents):
                results.append(contents[index])

        return results

    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return []

# Example Usage
if __name__ == "__main__":
    query = "What is a sample document?"
    results = query_database(query)
    print("\nQuery Results:")
    for result in results:
        print(result)
