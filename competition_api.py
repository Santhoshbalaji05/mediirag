from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import shutil
from src.rag_system import PDFNotesRAG
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hack-A-Cure Medical RAG API",
    description="API for medical Q&A system using RAG",
    version="1.0.0"
)

rag_system = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

def initialize_rag_system():
    """Initialize RAG system with existing vector store if available"""
    global rag_system
    try:
        logger.info("🚀 Initializing Hack-A-Cure RAG System...")
        
        rag_system = PDFNotesRAG("./data")
        
        # Check if vector store already exists and is valid
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path) and any(not f.startswith('.') for f in os.listdir(chroma_path)):
            logger.info("📂 Using existing Chroma DB...")
            try:
                vector_store = rag_system.setup_vector_store(force_recreate=False)
                logger.info("✅ Existing vector store loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing vector store: {e}")
                logger.info("🔄 Creating new vector store...")
                vector_store = rag_system.setup_vector_store(force_recreate=True)
        else:
            logger.info("🛠️ Creating new vector store...")
            # Load PDFs and create chunks first
            rag_system.load_pdfs()
            rag_system.chunk_documents()
            vector_store = rag_system.setup_vector_store(force_recreate=True)
        
        # Setup Gemini LLM
        api_key = "AIzaSyBpCOIHt6VO-OVj9pN8_PZC6oKtvlE14FI"
        success = rag_system.setup_gemini_llm(api_key)
        if success:
            logger.info("✅ Gemini LLM initialized")
        else:
            logger.warning("⚠️ Gemini LLM setup failed")
        
        # Quick test
        logger.info("🧪 Quick system test...")
        test_result = rag_system.ask_question("test", k=1)
        logger.info(f"✅ System test passed - Answer length: {len(test_result.get('answer', ''))}")
        
        logger.info("🎯 RAG System ready for queries!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG System initialization failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    import threading
    thread = threading.Thread(target=initialize_rag_system)
    thread.daemon = True
    thread.start()

@app.get("/")
async def root():
    status = "initializing" if rag_system is None else "ready"
    return {
        "message": "Hack-A-Cure Medical RAG API", 
        "status": status,
        "endpoint": "POST /query"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_system and rag_system.vector_store:
        return {
            "status": "healthy", 
            "rag_ready": True,
            "vector_store_ready": rag_system.vector_store is not None
        }
    return {"status": "initializing", "rag_ready": False}

@app.post("/query", response_model=QueryResponse)
async def query_medical_ai(request: QueryRequest):
    """Main competition endpoint"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system initializing, please wait...")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        result = rag_system.ask_question(
            question=request.query,
            conversation_context=[],
            explanation_mode="technical",
            k=request.top_k
        )
        
        # Convert to competition format
        contexts = []
        if result.get("sources"):
            for doc in result["sources"][:request.top_k]:
                context_text = doc.page_content.strip()
                if len(context_text) > 500:
                    context_text = context_text[:500] + "..."
                contexts.append(context_text)
        
        contexts = contexts[:request.top_k]
        
        return QueryResponse(
            answer=result.get("answer", "No answer generated."),
            contexts=contexts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)