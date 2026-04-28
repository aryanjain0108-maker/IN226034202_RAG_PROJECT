from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Use the EXACT same embedding model as ingest.py
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = "What is the refund policy?"
results = db.similarity_search(query, k=2)

print(f"Found {len(results)} matches:")
for i, res in enumerate(results):
    print(f"\nMatch {i+1} (from page {res.metadata.get('page')}):")
    print(res.page_content[:200] + "...")
