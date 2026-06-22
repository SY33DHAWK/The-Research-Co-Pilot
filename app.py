import os
import sys
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
import streamlit as st
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. INITIALIZATION & CLIENT SETUP
# ==========================================

# Load environment variables
load_dotenv()

# Setup Qwen client (using the working international endpoint)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Initialize ChromaDB for vector storage (Updated for ChromaDB v0.4+)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize embedding model (Downloads on first run, ~80MB)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def extract_text_from_pdf_with_metadata(pdf_file):
    """Extract text from PDF with page numbers and metadata using pdfplumber"""
    pages_text = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append({
                        "page": i + 1,
                        "text": page_text,
                        "filename": pdf_file.name
                    })
        return pages_text
    except Exception as e:
        st.error(f"Could not read {pdf_file.name}: {str(e)[:100]}")
        return []

def chunk_text(pages_text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks with truly unique citations"""
    chunks = []
    chunk_index = 0  # Global counter to guarantee uniqueness
    
    for page_data in pages_text:
        text = page_data["text"]
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            
            if len(chunk_str.strip()) > 50:  # Only keep meaningful chunks
                # Create a truly unique ID using filename, page, and index
                unique_id = f"{page_data['filename']}_p{page_data['page']}_c{chunk_index}"
                
                chunks.append({
                    "text": chunk_str,
                    "page": page_data["page"],
                    "filename": page_data["filename"],
                    "chunk_id": unique_id
                })
                chunk_index += 1
    
    return chunks

def create_or_get_collection(session_id):
    """Create or get ChromaDB collection for this session"""
    collection_name = f"papers_{session_id}"
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    return collection

def add_documents_to_vector_db(chunks, collection):
    """Add chunks to vector database safely"""
    if not chunks:
        return
    
    # Clear old data safely before adding new ones
    try:
        existing_data = collection.get()
        if existing_data and existing_data['ids']:
            collection.delete(ids=existing_data['ids'])
    except Exception as e:
        print(f"Note: Could not clear old collection data: {e}")

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts).tolist()
    
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"page": c["page"], "filename": c["filename"]} for c in chunks],
        ids=[c["chunk_id"] for c in chunks]
    )

def retrieve_relevant_chunks(query, collection, n_results=5):
    """Retrieve most relevant chunks for a query"""
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    # Format results with citations
    retrieved_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        retrieved_chunks.append({
            "text": doc,
            "page": metadata["page"],
            "filename": metadata["filename"]
        })
    
    return retrieved_chunks

def analyze_papers(papers_text, analysis_prompt):
    """Send full papers to Qwen for comprehensive analysis"""
    messages = [
        {
            'role': 'system',
            'content': 'You are an expert academic research assistant. Analyze the provided research papers and provide structured, insightful, and highly accurate responses. Use markdown formatting for tables and lists.'
        },
        {
            'role': 'user',
            'content': f"""
Here are the research papers:
{papers_text}

---
TASK: {analysis_prompt}

Provide your analysis in a clear, structured format.
"""
        }
    ]
    
    with st.spinner('Qwen is analyzing your papers... This might take a minute...'):
        response = client.chat.completions.create(
            model='qwen-plus',  # Using qwen-plus for 128k context window
            messages=messages,
            temperature=0.3
        )
    
    return response.choices[0].message.content

def answer_with_citations(query, retrieved_chunks):
    """Use Qwen to answer question based on retrieved chunks with citations"""
    context = "\n\n".join([
        f"[From {chunk['filename']}, Page {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])
    
    messages = [
        {
            'role': 'system',
            'content': (
                'You are an expert academic research assistant. '
                'Answer the user\'s question based ONLY on the provided context. '
                'Every claim MUST have a citation in the format [Paper Name, Page X]. '
                'If the answer is not in the context, say "I cannot find this information in the uploaded papers." '
                'Be precise and cite your sources.'
            )
        },
        {
            'role': 'user',
            'content': f"""
Context from research papers:
{context}

---
Question: {query}

Provide your answer with citations for every claim.
"""
        }
    ]
    
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="The Research Co-Pilot", page_icon="📚", layout="wide")

st.title("📚 The Research Co-Pilot")
st.markdown("Upload your research papers and **chat with them** to extract insights, compare methodologies, and find research gaps.")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'papers_loaded' not in st.session_state:
    st.session_state.papers_loaded = False
if 'papers_text' not in st.session_state:
    st.session_state.papers_text = ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    analysis_type = st.selectbox(
        "Quick Analysis",
        [
            "Research Matrix (Compare All Papers)",
            "Gap Radar (Find Contradictions & Gaps)",
            "Methodology Extractor",
            "Literature Gap Analysis",
            "Custom Prompt"
        ]
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# File uploader
uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    # Check if we need to process the files (prevents re-processing on every click)
    current_files = [f.name for f in uploaded_files]
    if st.session_state.get('last_uploaded') != current_files:
        st.session_state.papers_text = ""
        
        # Get or create collection
        collection = create_or_get_collection(st.session_state.session_id)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_chunks = []
        
        for i, pdf_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {pdf_file.name}")
            
            # Extract text with metadata
            pages_data = extract_text_from_pdf_with_metadata(pdf_file)
            
            # Chunk the text
            chunks = chunk_text(pages_data, chunk_size=500, overlap=50)
            all_chunks.extend(chunks)
            
            # Add text to papers_text for full analysis
            for page_data in pages_data:
                st.session_state.papers_text += f"\n\n=== {pdf_file.name} (Page {page_data['page']}) ===\n{page_data['text']}"
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Add all chunks to vector DB
        status_text.text("Building search index...")
        add_documents_to_vector_db(all_chunks, collection)
        
        st.session_state.papers_loaded = True
        st.session_state.last_uploaded = current_files
        
        status_text.empty()
        st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} papers!")
    
    # Tabs for different modes
    tab1, tab2 = st.tabs(["💬 Chat with Papers", "📊 Full Analysis"])
    
    with tab1:
        st.subheader("Ask questions about your research papers")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the papers..."):
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Retrieve relevant chunks
            collection = create_or_get_collection(st.session_state.session_id)
            retrieved_chunks = retrieve_relevant_chunks(prompt, collection, n_results=5)
            
            # Generate answer with citations
            with st.chat_message("assistant"):
                with st.spinner("Searching papers and generating answer..."):
                    answer = answer_with_citations(prompt, retrieved_chunks)
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander("📖 View Source Chunks"):
                        for chunk in retrieved_chunks:
                            st.markdown(f"**{chunk['filename']}, Page {chunk['page']}**")
                            st.text(chunk['text'][:300] + "...")
            
            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
    
    with tab2:
        st.subheader("Comprehensive Analysis")
        
        custom_prompt = ""
        if analysis_type == "Research Matrix (Compare All Papers)":
            custom_prompt = (
                "Create a comprehensive Research Matrix table comparing ALL uploaded papers. "
                "The table must have these columns: "
                "| Paper (Author, Year) | Dataset Used | Sample Size | Algorithms Tested | "
                "Best Algorithm | Best Accuracy | Feature Selection Method | "
                "Cross-Validation | Key Limitation |. "
                "After the table, write a 2-paragraph synthesis highlighting which approaches "
                "dominate the field and where the gaps are."
            )
        elif analysis_type == "Gap Radar (Find Contradictions & Gaps)":
            custom_prompt = (
                "Act as a critical research reviewer. Analyze all uploaded papers and produce:\n\n"
                "1. **Contradiction Map**: List every instance where two papers disagree "
                "(e.g., Paper A says Algorithm X is best, Paper B says Algorithm Y is best). "
                "Explain WHY they might disagree (different datasets? different preprocessing?).\n\n"
                "2. **White Space Map**: Identify what NO paper has done yet. "
                "What datasets are underexplored? What algorithms have not been tried? "
                "What combinations of techniques are missing?\n\n"
                "3. **Novel Thesis Suggestions**: Based on the gaps, suggest 3 specific, "
                "novel thesis research questions that a student could pursue."
            )
        elif analysis_type == "Methodology Extractor":
            custom_prompt = (
                "Extract ONLY the experimental methodology details from each paper. "
                "For each paper, create a structured card with:\n"
                "- **Paper**: Author, Year\n"
                "- **Dataset**: Name, source, number of records, number of features\n"
                "- **Preprocessing**: Missing value handling, normalization, SMOTE/oversampling\n"
                "- **Feature Selection**: Method used (LASSO, Wrapper, Filter, etc.), "
                "number of features selected\n"
                "- **Algorithms**: List all models tested\n"
                "- **Validation**: k-fold value, train/test split ratio\n"
                "- **Metrics Reported**: Accuracy, Precision, Recall, F1, AUC\n"
                "- **Tools/Libraries**: WEKA, Python sklearn, SPSS, etc.\n\n"
                "After all cards, create a summary table comparing preprocessing and "
                "feature selection choices across papers."
            )
        elif analysis_type == "Literature Gap Analysis":
            custom_prompt = (
                "Identify research gaps across these papers. What questions remain unanswered? "
                "What methodologies are missing? Suggest 3 potential research directions."
            )
        else:
            custom_prompt = st.text_area("Enter your custom prompt:", height=150)
        
        if st.button("🚀 Run Full Analysis", type="primary"):
            if custom_prompt:
                result = analyze_papers(st.session_state.papers_text, custom_prompt)
                
                st.subheader("📊 Analysis Results")
                st.markdown(result)
                
                st.download_button(
                    label="📥 Download Results as Markdown",
                    data=result,
                    file_name="thesis_analysis_results.md",
                    mime="text/markdown"
                )
else:
    st.info("👈 Please upload one or more PDF research papers to get started.")