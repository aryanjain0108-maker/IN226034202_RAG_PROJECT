from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the NovaTech PDF
loader = PyPDFLoader(r"C:/Users/aryan/Downloads/NovaTech_Customer_Support_KnowledgeBase.pdf")
docs = loader.load()

# 2. Optimized Chunking Strategy
# 1000 size captures full sections; 200 overlap prevents losing context at the edges
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True 
)
chunks = text_splitter.split_documents(docs)

# 3. Use a consistent Embedding Model (Free/Local)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 4. Create the new database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Success! Created {len(chunks)} optimized chunks.")
# Diagnostic: Try a direct search immediately after creating the DB
test_query = "What is NovaTech?"
found_docs = vectorstore.similarity_search(test_query, k=1)

if len(found_docs) > 0:
    print(f"DATABASE VERIFIED: Found content: {found_docs[0].page_content[:100]}...")
else:
    print("DATABASE ERROR: No content found even after ingestion.")
