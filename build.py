from download_chroma import download_chroma_db
from src.rag_system import PDFNotesRAG
import os

def setup_system():
    print("🚀 Setting up system with pre-built Chroma DB...")
    
    # Download existing Chroma DB
    download_chroma_db()
    
    # Initialize RAG (no need to process PDFs)
    rag = PDFNotesRAG("./data")  # Data folder can be empty
    
    # Load existing vector store
    rag.setup_vector_store(force_recreate=False)  # FALSE = use existing
    
    # Setup LLM only
    api_key = "AIzaSyBpCOIHt6VO-OVj9pN8_PZC6oKtvlE14FI"
    rag.setup_gemini_llm(api_key)
    
    print("✅ System ready in 2 minutes!")

if __name__ == "__main__":
    setup_system()
