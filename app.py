import os
import streamlit as st
from src.rag_system import PDFNotesRAG
import requests
import json

st.set_page_config(
    page_title="Medical Assistant Pro",
    page_icon="⚕️",
    layout="centered"
)

st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .terminal-output {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.4;
        margin-bottom: 10px;
    }
    .user-text {
        color: #00ff00;
        font-weight: bold;
    }
    .assistant-text {
        color: #ffffff;
    }
    .source-text {
        color: #888888;
        font-size: 12px;
        margin-left: 20px;
    }
    .sources-container {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 3px solid #00ff00;
    }
    .source-item {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #cccccc;
        margin: 2px 0;
    }
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
    .api-test-success { color: #4CAF50; font-weight: bold; }
    .api-test-fail { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

def test_api_connection():
    """Test connection to the competition API"""
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"Status code: {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def query_api(question, top_k=3):
    """Query the competition API"""
    try:
        payload = {
            "query": question,
            "top_k": top_k
        }
        response = requests.post(
            f"{st.session_state.api_url}/query",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

# Sidebar for settings and API testing
with st.sidebar:
    st.markdown("### ⚙️ Competition API Settings")
    
    # API URL configuration
    api_url = st.text_input(
        "API Base URL:",
        value=st.session_state.api_url,
        help="URL where your competition API is running"
    )
    st.session_state.api_url = api_url
    
    # Test API connection
    if st.button("🔗 Test API Connection"):
        with st.spinner("Testing connection..."):
            success, result = test_api_connection()
            if success:
                st.markdown(f'<p class="api-test-success">✅ API is healthy!</p>', unsafe_allow_html=True)
                st.json(result)
            else:
                st.markdown(f'<p class="api-test-fail">❌ API connection failed</p>', unsafe_allow_html=True)
                st.error(result.get("error", "Unknown error"))
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    - ✅ Competition API Compatible
    - ✅ Citation-Backed Answers  
    - ✅ Confidence Scoring
    - ✅ Dual Explanation Modes
    - ✅ Conversation Memory
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Competition Info")
    st.markdown("""
    **Endpoint:** `POST /query`
    
    **Request:**
    ```json
    {
      "query": "medical question",
      "top_k": 3
    }
    ```
    
    **Response:**
    ```json
    {
      "answer": "medical answer",
      "contexts": ["source1", "source2"]
    }
    ```
    """)

st.markdown("### 🏥 Medical Assistant Pro - Competition Demo")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="terminal-output"><span class="user-text">You:</span> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        # Display answer
        st.markdown(f'<div class="terminal-output"><span class="assistant-text">Assistant:</span> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Display confidence score if available
        if message.get("confidence"):
            confidence = message["confidence"]
            if confidence > 70:
                confidence_class = "confidence-high"
                emoji = "🟢"
            elif confidence > 40:
                confidence_class = "confidence-medium" 
                emoji = "🟡"
            else:
                confidence_class = "confidence-low"
                emoji = "🔴"
            
            st.markdown(f'<div class="source-text">{emoji} Confidence: <span class="{confidence_class}">{confidence:.1f}%</span></div>', unsafe_allow_html=True)
        
        # Display sources in clean format
        if message.get("sources"):
            sources_html = '<div class="sources-container">'
            sources_html += '<div style="color: #00ff00; font-size: 12px; margin-bottom: 8px;">📚 Sources:</div>'
            
            for i, source in enumerate(message["sources"][:5]):  # Limit to 5 sources
                unit = source.metadata.get('unit', 'Unknown')
                page = source.metadata.get('page', 0) + 1
                sources_html += f'<div class="source-item">• {unit} : Page {page}</div>'
            
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)
        
        # Display API contexts if available
        if message.get("api_contexts"):
            contexts_html = '<div class="sources-container">'
            contexts_html += '<div style="color: #00ff00; font-size: 12px; margin-bottom: 8px;">🔍 API Contexts:</div>'
            
            for i, context in enumerate(message["api_contexts"][:3]):
                preview = context[:100] + "..." if len(context) > 100 else context
                contexts_html += f'<div class="source-item">[{i+1}] {preview}</div>'
            
            contexts_html += '</div>'
            st.markdown(contexts_html, unsafe_allow_html=True)

st.markdown("---")

# Input area
st.markdown("### 💬 Ask Medical Questions")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    prompt = st.text_input(
        "Enter your question:",
        key="user_input",
        label_visibility="collapsed",
        placeholder="Type your medical question here..."
    )
with col2:
    use_api = st.checkbox("Use API", value=False, help="Use competition API instead of local RAG")
with col3:
    send_clicked = st.button("🚀 Send")

clear_clicked = st.button("🗑️ Clear Chat History")

if clear_clicked:
    st.session_state.messages = []
    st.rerun()

if send_clicked and prompt.strip():
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if use_api:
        # Use competition API
        with st.spinner("🔍 Querying competition API..."):
            success, result = query_api(prompt, top_k=3)
            
            if success:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result.get("answer", "No answer from API."),
                    "api_contexts": result.get("contexts", [])
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"API Error: {result.get('error', 'Unknown error')}"
                })
    else:
        # Use local RAG system (demo mode)
        with st.spinner("🔍 Analyzing with local RAG system..."):
            try:
                # Initialize RAG if needed
                if "rag" not in st.session_state:
                    with st.spinner("🚀 Initializing RAG system..."):
                        rag = PDFNotesRAG("./data")
                        if os.path.exists("./chroma_db") and any(os.listdir("./chroma_db")):
                            rag.setup_vector_store(force_recreate=False)
                        else:
                            rag.setup_vector_store(force_recreate=True)
                        api_key = "AIzaSyBpCOIHt6VO-OVj9pN8_PZC6oKtvlE14FI"
                        rag.setup_gemini_llm(api_key)
                        st.session_state.rag = rag
                
                result = st.session_state.rag.ask_question(
                    prompt, 
                    conversation_context=[],
                    explanation_mode="technical"
                )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result.get("answer", "No answer found."),
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0)
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Error: {str(e)}"
                })
    
    st.rerun()

# Footer with competition info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    <strong>Hack-A-Cure 2025 Competition Ready</strong><br>
    Local Demo Mode | API Competition Mode
</div>
""", unsafe_allow_html=True)