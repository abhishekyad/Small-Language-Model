from langchain.embeddings import HuggingFaceEmbeddings

# Use a lightweight model that runs smoothly on Mac
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
