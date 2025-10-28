import os
import time
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import google.generativeai as genai
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class PDFNotesRAG:
    def __init__(self, notes_directory, persist_directory="./chroma_db"):
        self.notes_directory = notes_directory
        self.persist_directory = persist_directory
        self.documents = []
        self.chunks = []
        self.vector_store = None
        self.embeddings = None
        self.genai_model = None
        self._cache = {}
        self.cache_lock = threading.Lock()
        self.current_docs = []
        
        # Ensure data directory exists
        os.makedirs(self.notes_directory, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)

    def load_pdfs(self) -> List[Document]:
        """Load all PDF files with robust error handling and parallel processing"""
        cache_key = "loaded_pdfs"
        with self.cache_lock:
            if cache_key in self._cache:
                self.documents = self._cache[cache_key]
                return self.documents
        
        logger.info("📚 Loading PDF documents...")
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(self.notes_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"❌ No PDF files found in {self.notes_directory}")
            return []
        
        logger.info(f"📖 Found {len(pdf_files)} PDF files: {pdf_files}")

        def load_single_pdf(filename):
            """Load a single PDF file with error handling"""
            file_path = os.path.join(self.notes_directory, filename)
            try:
                logger.info(f"📄 Loading {filename}...")
                
                # Try PyPDFLoader first (faster)
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                
                # Enhanced metadata
                for i, doc in enumerate(pdf_documents):
                    doc.metadata.update({
                        'unit': filename.replace('.pdf', '').replace('.PDF', ''),
                        'source': filename,
                        'page': i,
                        'quality_score': self._calculate_document_quality(doc.page_content),
                        'file_size': os.path.getsize(file_path),
                        'total_pages': len(pdf_documents)
                    })
                
                logger.info(f"✅ Successfully loaded {filename} - {len(pdf_documents)} pages, {sum(len(d.page_content) for d in pdf_documents)} chars")
                return pdf_documents
                
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {str(e)}")
                try:
                    # Fallback to UnstructuredPDFLoader
                    logger.info(f"🔄 Trying fallback loader for {filename}...")
                    loader = UnstructuredPDFLoader(file_path)
                    pdf_documents = loader.load()
                    logger.info(f"✅ Fallback successful for {filename}")
                    return pdf_documents
                except Exception as e2:
                    logger.error(f"❌ Fallback also failed for {filename}: {str(e2)}")
                    return []

        # Parallel loading
        all_documents = []
        with ThreadPoolExecutor(max_workers=min(4, len(pdf_files))) as executor:
            future_to_file = {executor.submit(load_single_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
            
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    pdf_docs = future.result()
                    if pdf_docs:
                        all_documents.extend(pdf_docs)
                        total_chars = sum(len(doc.page_content) for doc in pdf_docs)
                        logger.info(f"✅ Completed {pdf_file}: {len(pdf_docs)} pages, {total_chars} chars")
                    else:
                        logger.warning(f"⚠️ No documents extracted from {pdf_file}")
                except Exception as e:
                    logger.error(f"❌ Error processing {pdf_file}: {str(e)}")

        self.documents = all_documents
        
        # Statistics
        if self.documents:
            total_chars = sum(len(doc.page_content) for doc in self.documents)
            avg_chars = total_chars / len(self.documents)
            logger.info(f"📊 Loaded {len(self.documents)} pages from {len(pdf_files)} files")
            logger.info(f"📊 Total characters: {total_chars:,}, Average per page: {avg_chars:.0f}")
            
            # Log document sources
            sources = set(doc.metadata.get('unit', 'unknown') for doc in self.documents)
            logger.info(f"📚 Sources: {list(sources)}")
        else:
            logger.error("❌ No documents were successfully loaded!")
        
        with self.cache_lock:
            self._cache[cache_key] = self.documents
        
        return self.documents

    def _calculate_document_quality(self, content: str) -> float:
        """Calculate quality score for document content"""
        if not content or not content.strip():
            return 0.0
        
        score = 50.0  # Base score
        
        # Length scoring
        content_length = len(content.strip())
        if content_length > 1000:
            score += 20
        elif content_length > 500:
            score += 15
        elif content_length > 200:
            score += 10
        elif content_length > 50:
            score += 5
        else:
            score -= 10
        
        # Content quality indicators
        text = content.lower()
        
        # Medical terminology indicators
        medical_terms = ['patient', 'treatment', 'diagnosis', 'symptoms', 'therapy', 'clinical', 'medical', 
                        'disease', 'condition', 'medicine', 'drug', 'dose', 'mg', 'ml', 'blood', 'heart',
                        'cancer', 'diabetes', 'infection', 'virus', 'bacteria', 'surgery', 'procedure']
        
        medical_count = sum(1 for term in medical_terms if term in text)
        score += min(20, medical_count * 2)
        
        # Structure indicators
        if '\n\n' in content:
            score += 5
        if any(char.isdigit() for char in content):
            score += 5
        if any(marker in text for marker in ['introduction', 'conclusion', 'summary', 'abstract']):
            score += 10
        
        # Penalize very poor content
        if content_length < 30:
            score -= 20
        
        return min(100.0, max(0.0, score))

    def chunk_documents(self, chunk_size=800, chunk_overlap=150) -> List[Document]:
        """Split documents into chunks with medical-aware splitting"""
        cache_key = f"chunks_{chunk_size}_{chunk_overlap}"
        with self.cache_lock:
            if cache_key in self._cache:
                self.chunks = self._cache[cache_key]
                return self.chunks
        
        logger.info("✂️ Chunking documents...")
        
        if not self.documents:
            self.load_pdfs()
        
        if not self.documents:
            logger.error("❌ No documents to chunk!")
            return []
        
        # Use medical-aware text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n", 
                "\n", 
                ". ",
                "! ",
                "? ",
                "; ",
                ": ",
                " ",
                ""
            ]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        
        # Filter out very low quality chunks
        initial_count = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if self._calculate_document_quality(chunk.page_content) > 20]
        filtered_count = initial_count - len(self.chunks)
        
        if filtered_count > 0:
            logger.info(f"🚮 Filtered out {filtered_count} low-quality chunks")
        
        logger.info(f"✅ Created {len(self.chunks)} chunks from {len(self.documents)} pages")
        
        # Log chunk statistics
        if self.chunks:
            chunk_lengths = [len(chunk.page_content) for chunk in self.chunks]
            logger.info(f"📊 Chunk length stats: min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={np.mean(chunk_lengths):.0f}")
        
        with self.cache_lock:
            self._cache[cache_key] = self.chunks
        
        return self.chunks

    def setup_vector_store(self, force_recreate=False):
        """Create embeddings and vector store with robust error handling"""
        logger.info("🛠️ Setting up vector store...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Check if we should use existing vector store
        vector_db_exists = os.path.exists(self.persist_directory) and any(
            not f.startswith('.') for f in os.listdir(self.persist_directory)
        )
        
        if vector_db_exists and not force_recreate:
            logger.info("📂 Loading existing vector store...")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # Test the store
                test_results = self.vector_store.similarity_search("test", k=1)
                logger.info(f"✅ Vector DB loaded successfully - {self.vector_store._collection.count()} embeddings")
                return self.vector_store
            except Exception as e:
                logger.warning(f"⚠️ Error loading existing vector store: {e}")
                logger.info("🔄 Recreating vector store...")
                force_recreate = True
        
        # Create new vector store
        logger.info("🛠️ Creating new vector store...")
        
        if not self.chunks:
            self.chunk_documents()
        
        if not self.chunks:
            logger.error("❌ No chunks available for vector store creation!")
            raise ValueError("No document chunks available")
        
        logger.info(f"📊 Creating embeddings for {len(self.chunks)} chunks...")
        
        try:
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Verify creation
            count = self.vector_store._collection.count()
            logger.info(f"✅ Vector store created successfully - {count} embeddings persisted")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            raise

    def setup_gemini_llm(self, api_key):
        """Setup Google Gemini LLM with proper configuration"""
        try:
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Test the model
            test_response = self.genai_model.generate_content("Say 'MEDICAL AI READY' in one word.")
            
            if test_response and test_response.text:
                logger.info("✅ Google Gemini AI loaded successfully!")
                return True
            else:
                logger.error("❌ Gemini test response was empty")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting up Gemini: {e}")
            return False

    def _enhance_prompt_with_context(self, context: str, question: str, explanation_mode: str = "technical") -> str:
        """Create enhanced prompt for medical Q&A"""
        mode_instruction = "Use detailed medical terminology and be precise." if explanation_mode == "technical" else "Explain in simple, easy-to-understand language for patients."
        
        prompt = f"""You are a medical expert assistant. Answer the question based ONLY on the provided medical textbook content.

QUESTION: {question}

MEDICAL TEXTBOOK CONTEXT:
{context}

{mode_instruction}

CRITICAL INSTRUCTIONS:
1. Answer using ONLY the information from the medical textbook context above
2. If the information isn't in the context, say exactly: "I cannot find this information in the available medical textbooks."
3. Do NOT make up or hallucinate any information - be strictly factual
4. Be comprehensive but concise in your medical explanation
5. Include relevant medical details, terminology, and clinical information
6. DO NOT include citations or source references in your answer text
7. Provide a clean, readable medical answer without any [Source: X] formatting
8. If multiple relevant points are found, organize them clearly

MEDICAL ANSWER:"""
        
        return prompt

    def ask_question(self, question: str, conversation_context: List[str] = None, explanation_mode: str = "technical", k: int = 4) -> Dict[str, Any]:
        """Advanced question answering with medical focus"""
        if self.vector_store is None:
            return {"error": "Vector store not initialized"}
        
        start_time = time.time()
        
        logger.info(f"🔍 Processing question: '{question}'")
        
        # Step 1: Retrieve relevant documents - FIXED VERSION
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}  # REMOVED score_threshold
            )
            relevant_docs = retriever.invoke(question)
            
            if not relevant_docs:
                logger.warning("⚠️ No relevant documents found for query")
                return {
                    "question": question,
                    "answer": "I cannot find this information in the available medical textbooks.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            logger.info(f"📄 Retrieved {len(relevant_docs)} relevant documents")
            
        except Exception as e:
            logger.error(f"❌ Error during document retrieval: {e}")
            return {
                "question": question,
                "answer": f"Error retrieving information: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    # ... rest of the method remains the same ...
        
        # Step 2: Prepare context
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('unit', 'Unknown')
            page = doc.metadata.get('page', 0) + 1
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"[Source: {source}, Page: {page}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer using LLM
        if self.genai_model:
            try:
                enhanced_prompt = self._enhance_prompt_with_context(context, question, explanation_mode)
                
                response = self.genai_model.generate_content(enhanced_prompt)
                
                if response and response.text:
                    answer_text = response.text.strip()
                    
                    # Calculate confidence based on relevance
                    confidence = self._calculate_confidence(question, relevant_docs, answer_text)
                    
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Answer generated in {processing_time:.2f}s | Confidence: {confidence:.1f}%")
                    
                    return {
                        "question": question,
                        "answer": answer_text,
                        "sources": relevant_docs,
                        "confidence": confidence
                    }
                else:
                    logger.warning("⚠️ LLM returned empty response")
                    return {
                        "question": question,
                        "answer": "I couldn't generate a proper answer. Here's the relevant medical context:\n\n" + context,
                        "sources": relevant_docs,
                        "confidence": 40.0
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error during LLM generation: {e}")
                return {
                    "question": question,
                    "answer": f"Error generating answer: {str(e)}\n\nRelevant medical context:\n{context}",
                    "sources": relevant_docs,
                    "confidence": 30.0
                }
        else:
            # Fallback: return context directly
            logger.warning("⚠️ Using fallback mode (no LLM)")
            return {
                "question": question,
                "answer": f"Relevant medical information from textbooks:\n\n{context}",
                "sources": relevant_docs,
                "confidence": 50.0
            }

    def _calculate_confidence(self, question: str, documents: List[Document], answer: str) -> float:
        """Calculate confidence score based on multiple factors"""
        if not documents:
            return 0.0
        
        scores = []
        
        # 1. Answer presence and quality
        if "cannot find" in answer.lower():
            scores.append(10.0)
        elif len(answer.strip()) < 20:
            scores.append(20.0)
        else:
            scores.append(70.0)
        
        # 2. Document relevance (simple keyword matching)
        question_terms = set(question.lower().split())
        common_terms = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'what', 'how', 'when', 'where', 'why'}
        question_terms = question_terms - common_terms
        
        if question_terms:
            doc_scores = []
            for doc in documents:
                doc_text = doc.page_content.lower()
                matches = sum(1 for term in question_terms if term in doc_text)
                doc_scores.append(min(100.0, (matches / len(question_terms)) * 100))
            
            if doc_scores:
                scores.append(np.mean(doc_scores))
        
        # 3. Document quality scores
        quality_scores = [doc.metadata.get('quality_score', 50) for doc in documents]
        if quality_scores:
            scores.append(np.mean(quality_scores))
        
        final_confidence = np.mean(scores) if scores else 50.0
        return min(100.0, max(0.0, final_confidence))

    def debug_system(self):
        """Debug method to check system status"""
        logger.info("🔍 Debugging RAG System...")
        
        logger.info(f"📊 Documents loaded: {len(self.documents)}")
        logger.info(f"📊 Chunks created: {len(self.chunks)}")
        logger.info(f"📊 Vector store ready: {self.vector_store is not None}")
        logger.info(f"📊 LLM ready: {self.genai_model is not None}")
        
        if self.documents:
            # Show sample of documents
            logger.info("📄 Sample documents:")
            for i, doc in enumerate(self.documents[:2]):
                logger.info(f"  {i+1}. {doc.metadata.get('unit', 'unknown')} - Page {doc.metadata.get('page', 0)} - {len(doc.page_content)} chars")
        
        if self.vector_store:
            try:
                count = self.vector_store._collection.count()
                logger.info(f"📊 Vector store embeddings: {count}")
            except:
                logger.info("📊 Vector store count unavailable")

if __name__ == "__main__":
    # Test the system
    rag = PDFNotesRAG("./data")
    
    # Force fresh setup
    rag.load_pdfs()
    rag.chunk_documents()
    rag.setup_vector_store(force_recreate=True)
    rag.setup_gemini_llm("YOUR_API_KEY_HERE")
    
    rag.debug_system()
    
    # Test queries
    test_queries = [
        "What is diabetes?",
        "How to treat hypertension?",
        "What are the symptoms of COVID-19?",
        "Explain heart anatomy",
        "What is Tdap booster?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"❓ Query: {query}")
        result = rag.ask_question(query)
        print(f"🤖 Answer: {result['answer'][:200]}...")
        print(f"🎯 Confidence: {result.get('confidence', 0):.1f}%")

        print(f"📚 Sources: {len(result.get('sources', []))}")
