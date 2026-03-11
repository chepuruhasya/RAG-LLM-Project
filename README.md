
# RAG-LLM Project

This project implements Retrieval Augmented Generation (RAG).

Tech Stack:
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Streamlit UI

Pipeline:
User Query → Embedding → Vector Search → LLM → Answer

Steps:
1. Add a PDF inside the data/ folder
2. Run: python create_vector_db.py
3. Start UI: streamlit run app.py
