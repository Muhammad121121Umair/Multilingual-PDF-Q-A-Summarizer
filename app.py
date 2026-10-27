import os
import re
import io
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

MODEL_ID = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="Multilingual PDF RAG (English & Urdu)",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------------
# Helper Functions: PDF Extraction & Chunking
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text across all pages in the PDF."""
    reader = PdfReader(uploaded_file)
    extracted_text = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(page_text)
    return "\n".join(extracted_text)

def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[str]:
    """Splits text into overlapping chunks, respecting English and Urdu delimiters."""
    # Split across paragraph breaks, English sentence ends, or Urdu sentence markers (۔ / ؟)
    paragraphs = re.split(r'(\n\n+|(?<=[.!?۔؟])\s+)', text)
    chunks = []
    current_chunk = ""

    for segment in paragraphs:
        if not segment.strip():
            continue
        if len(current_chunk) + len(segment) <= chunk_size:
            current_chunk += " " + segment.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Maintain sliding overlap
            overlap_prefix = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
            current_chunk = (overlap_prefix + " " + segment.strip()).strip()

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [c for c in chunks if len(c.strip()) > 30]

# ---------------------------------------------------------
# Vector Indexing & Retrieval Engine
# ---------------------------------------------------------
class MultilingualRetriever:
    """Character n-gram vectorizer for robust English and Nastaliq/Urdu retrieval."""
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[idx] for idx in top_indices if similarities[idx] > 0.05]

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "full_doc_text" not in st.session_state:
    st.session_state.full_doc_text = ""

# ---------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Reads GROQ_API_KEY from environment or manual entry."
    )

    st.markdown("---")
    st.subheader("📄 Upload Document")
    uploaded_pdf = st.file_uploader("Upload PDF (English or Urdu)", type=["pdf"])

    if uploaded_pdf and st.button("Process Document", type="primary", use_container_width=True):
        with st.spinner("Extracting and building multilingual index..."):
            raw_text = extract_text_from_pdf(uploaded_pdf)
            if not raw_text.strip():
                st.error("No extractable text found in this PDF (it may contain scanned image pages).")
            else:
                chunks = chunk_text(raw_text)
                st.session_state.chunks = chunks
                st.session_state.full_doc_text = raw_text
                st.session_state.retriever = MultilingualRetriever(chunks)
                st.session_state.chat_history = []
                st.success(f"Indexed {len(chunks)} chunks successfully!")

    if st.session_state.chunks:
        st.info(f"Active Index: **{len(st.session_state.chunks)} text chunks**")
        if st.button("Clear Document & Index", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.session_state.full_doc_text = ""
            st.rerun()

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.header("📚 Multilingual PDF Q&A & Summarizer")
st.caption("Ask questions or summarize documents in **English** or **Urdu (اردو)** using Groq.")

if not st.session_state.chunks:
    st.info("👈 Please upload a PDF and click **Process Document** in the sidebar to begin.")
    st.stop()

# Quick Action: Document Summary
with st.expander("📝 Generate Instant Summary / دستاویز کا خلاصہ", expanded=False):
    summary_lang = st.radio("Summary Language / خلاصہ کی زبان:", ["English", "Urdu (اردو)"], horizontal=True)
    if st.button("Generate Summary"):
        if not api_key:
            st.error("Please provide a Groq API Key.")
            st.stop()

        groq_client = Groq(api_key=api_key)
        
        # Take a representative window of text
        context_preview = st.session_state.full_doc_text[:12000]
        
        sum_prompt = (
            "Provide an executive summary of the provided text with key takeaways, main findings, and action points. "
            f"Respond strictly in {'Urdu (اردو)' if 'Urdu' in summary_lang else 'English'}.\n\n"
            f"DOCUMENT CONTENT:\n{context_preview}"
        )

        with st.spinner("Generating summary..."):
            try:
                res = groq_client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": "You are a professional multilingual document analyst."},
                        {"role": "user", "content": sum_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                )
                st.markdown(res.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# Render past chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Question Input
user_query = st.chat_input("Ask a question about the document (in English or اردو)...")

if user_query:
    if not api_key:
        st.error("Please enter a Groq API Key in the sidebar.")
        st.stop()

    # Display user query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve relevant chunks
    relevant_chunks = st.session_state.retriever.retrieve(user_query, top_k=4)
    if not relevant_chunks:
        # Fallback to initial chunks if query match score is low
        relevant_chunks = st.session_state.chunks[:3]

    context_str = "\n\n---\n\n".join(relevant_chunks)

    system_instruction = (
        "You are an expert bilingual research assistant and document auditor.\n"
        "Your task is to answer the user's question based strictly on the provided context.\n\n"
        "STRICT GUIDELINES:\n"
        "1. Language Matching: If the user asks in Urdu (اردو), reply in fluent, grammatically accurate Urdu. "
        "If the user asks in English, reply in English.\n"
        "2. Grounding: Answer ONLY from the retrieved context. If the information is not contained in the text, "
        "state clearly in the target language that the document does not contain that information.\n"
        "3. Clarity: Structure answers cleanly using bullet points or concise paragraphs."
    )

    prompt_payload = (
        f"RETRIEVED DOCUMENT CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        "Answer the question using the context above:"
    )

    groq_client = Groq(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Searching document context and generating answer..."):
            try:
                response = groq_client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt_payload}
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                with st.expander("🔍 View Retrieved Context Chunks"):
                    for idx, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {idx}:**\n{chunk}")

            except Exception as e:
                st.error(f"Groq API Error: {e}")
