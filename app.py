import os
import json
import math
import re
from collections import Counter
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
STORE_FILE = "knowledge_base.json"
MODEL_ID = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="AI Style Humanizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Pure-Python Vector & Similarity Engine (Zero C++ dependency)
# ---------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def compute_tfidf_vector(tokens: list[str], idf: dict) -> dict:
    tf = Counter(tokens)
    total_tokens = len(tokens) or 1
    vector = {}
    for word, count in tf.items():
        if word in idf:
            vector[word] = (count / total_tokens) * idf[word]
    return vector

def cosine_similarity(v1: dict, v2: dict) -> float:
    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# ---------------------------------------------------------
# Local Persistent Storage
# ---------------------------------------------------------
def load_knowledge_base() -> dict:
    if os.path.exists(STORE_FILE):
        try:
            with open(STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"chunks": [], "style_blueprint": ""}
    return {"chunks": [], "style_blueprint": ""}

def save_knowledge_base(data: dict):
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def retrieve_top_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    if not chunks:
        return []
    
    tokenized_corpus = [tokenize(c) for c in chunks]
    num_docs = len(tokenized_corpus)
    
    # Compute IDF
    doc_freq = Counter()
    for doc in tokenized_corpus:
        doc_freq.update(set(doc))
    
    idf = {w: math.log((num_docs + 1) / (df + 1)) + 1 for w, df in doc_freq.items()}
    
    query_tokens = tokenize(query)
    q_vec = compute_tfidf_vector(query_tokens, idf)
    
    scored = []
    for i, doc_tokens in enumerate(tokenized_corpus):
        d_vec = compute_tfidf_vector(doc_tokens, idf)
        score = cosine_similarity(q_vec, d_vec)
        scored.append((score, chunks[i]))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k] if score > 0] or chunks[:top_k]

# ---------------------------------------------------------
# Document Parsers
# ---------------------------------------------------------
def extract_text(file) -> str:
    name = file.name.lower()
    text = ""
    try:
        if name.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif name.endswith(".docx"):
            doc = Document(file)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return text.strip()

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def extract_style_blueprint(groq_client: Groq, sample_texts: list[str]) -> str:
    combined = "\n---\n".join(sample_texts[:5])
    prompt = (
        "Analyze these writing samples from a single writer. "
        "Formulate a concise, bulleted 'Writer Persona & Style Guide'. "
        "Detail:\n"
        "1. Sentence structure, cadence, and typical length variations.\n"
        "2. Vocabulary, phrase choices, transitions, and idioms.\n"
        "3. Tone, formality, and rhetorical flow.\n\n"
        f"WRITING SAMPLES:\n{combined[:4000]}\n\n"
        "Provide only the crisp stylistic blueprint."
    )
    response = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are an expert linguistic analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
kb_data = load_knowledge_base()

with st.sidebar:
    st.title("⚙️ Knowledge Base")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Enter your gsk_... key from console.groq.com"
    )

    st.markdown("---")
    st.subheader("📚 Upload Writing Samples")
    uploaded_files = st.file_uploader(
        "Upload past writing (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    process_btn = col1.button("Ingest Samples", use_container_width=True)
    clear_btn = col2.button("Clear Index", type="secondary", use_container_width=True)

    if clear_btn:
        save_knowledge_base({"chunks": [], "style_blueprint": ""})
        st.success("Knowledge base cleared.")
        st.rerun()

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Indexing samples and analyzing style..."):
                all_chunks = list(kb_data.get("chunks", []))
                previews = []

                for f in uploaded_files:
                    txt = extract_text(f)
                    if txt:
                        previews.append(txt[:1000])
                        chunks = chunk_text(txt)
                        all_chunks.extend(chunks)

                blueprint = kb_data.get("style_blueprint", "")
                if api_key and previews:
                    try:
                        client = Groq(api_key=api_key)
                        blueprint = extract_style_blueprint(client, previews)
                    except Exception as e:
                        st.error(f"Groq error during style analysis: {e}")

                save_knowledge_base({
                    "chunks": all_chunks,
                    "style_blueprint": blueprint
                })
                st.success(f"Indexed {len(all_chunks)} total chunks successfully!")
                st.rerun()

    st.markdown("---")
    st.subheader("📊 Profile Status")
    st.write(f"**Stored Chunks:** {len(kb_data.get('chunks', []))}")
    if kb_data.get("style_blueprint"):
        with st.expander("View Active Style Blueprint"):
            st.write(kb_data["style_blueprint"])
    else:
        st.info("Ingest samples with your API key to generate your style blueprint.")

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.header("✍️ AI Content Humanizer (Personalized RAG)")
st.caption(f"Powered by `{MODEL_ID}` on Groq with vector-matched personal voice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "raw_text" in msg:
            st.download_button(
                label="📥 Download Rewrite (.txt)",
                data=msg["raw_text"],
                file_name="humanized_output.txt",
                mime="text/plain",
                key=msg["id"]
            )

if user_prompt := st.chat_input("Paste draft to rewrite in your authentic voice..."):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    groq_client = Groq(api_key=api_key)
    
    # RAG Retrieval
    chunks = kb_data.get("chunks", [])
    matched_chunks = retrieve_top_chunks(user_prompt, chunks, top_k=3)
    ref_samples = "\n---\n".join(matched_chunks) if matched_chunks else "No uploaded writing samples found."
    blueprint = kb_data.get("style_blueprint", "Write clearly, directly, and naturally without generic AI phrases.")

    system_prompt = (
        "You are an expert personal ghostwriter. Rewrite the user's text so that it mirrors "
        "their exact personal writing voice, cadence, and sentence patterns, while retaining "
        "100% of facts and original meaning.\n\n"
        "DIRECTIVES:\n"
        "- Eliminate generic AI tropes ('delve', 'tapestry', 'testament', 'revolutionize', 'in conclusion').\n"
        "- Emulate the pacing, rhythm, and phrasing of the reference samples.\n"
        "- Output ONLY the rewritten text without preambles.\n\n"
        f"=== WRITER STYLE BLUEPRINT ===\n{blueprint}\n\n"
        f"=== RETRIEVED VOICE SAMPLES ===\n{ref_samples}"
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        try:
            stream = groq_client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Rewrite in my voice:\n\n{user_prompt}"}
                ],
                temperature=0.6,
                max_tokens=3000,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_text += token
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)

            msg_id = f"msg_{len(st.session_state.messages)}"
            st.download_button(
                label="📥 Download Rewrite (.txt)",
                data=full_text,
                file_name="humanized_output.txt",
                mime="text/plain",
                key=msg_id
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "raw_text": full_text,
                "id": msg_id
            })
        except Exception as e:
            st.error(f"Groq API Error: {e}")