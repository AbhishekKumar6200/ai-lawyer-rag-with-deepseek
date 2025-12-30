import streamlit as st
import numpy as np

from langchain_community.document_loaders import PDFPlumberLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore 
from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS  # type: ignore
from groq import Groq, BadRequestError
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from pyvis.network import Network
from rank_bm25 import BM25Okapi
import base64
import urllib.parse



custom_prompt_template = """
You are an AI Legal Assistant.
Use the pieces of information provided in the context and the conversation summary to answer the user's question.
If you don't know the answer, say you don't know. Do not invent facts.
Do not provide anything outside the given context.

Conversation Summary:
{memory}

Question:
{question}

Context:
{context}

Answer:
"""

OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
FAISS_DB_ROOT = "vectorstore/db_faiss"
MANIFEST_PATH = os.path.join(FAISS_DB_ROOT, "manifest.json")
PDFS_DIR = "pdfs/"

st.set_page_config(page_title="AI Lawyer RAG", page_icon="⚖️", layout="wide")
top_k = st.slider("Max source chunks (k)", min_value=2, max_value=10, value=5)
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Please set it in environment or .env file.")
groq_client = Groq(api_key=groq_api_key)

def resolve_groq_model(client: Groq) -> str:
    env_model = os.environ.get("GROQ_MODEL")
    if env_model:
        return env_model
    try:
        models = client.models.list()
        available_ids = [m.id for m in getattr(models, "data", [])]
        preferred = os.environ.get(
            "GROQ_MODEL_PREFERENCE",
            "llama-3.3-70b-versatile, llama-3.2-11b-text-preview, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it",
        )
        prefs = [m.strip() for m in preferred.split(",") if m.strip()]
        for p in prefs:
            if p in available_ids:
                return p
        for mid in available_ids:
            if "llama" in mid:
                return mid
        if available_ids:
            return available_ids[0]
    except Exception:
        # If listing models fails, return a conservative default that often exists.
        pass
    return "llama-3.2-11b-text-preview"

GROQ_MODEL = resolve_groq_model(groq_client)

def _ensure_dirs():
    os.makedirs(PDFS_DIR, exist_ok=True)
    os.makedirs(FAISS_DB_ROOT, exist_ok=True)

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_manifest() -> List[Dict]:
    _ensure_dirs()
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_manifest(entries: List[Dict]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

def upsert_manifest_entry(entry: Dict):
    entries = load_manifest()
    # Remove any existing entry with same doc_id
    entries = [e for e in entries if e.get("doc_id") != entry.get("doc_id")]
    entries.append(entry)
    save_manifest(entries)

def delete_manifest_entry(doc_id: str):
    entries = load_manifest()
    entries = [e for e in entries if e.get("doc_id") != doc_id]
    save_manifest(entries)

def upload_pdf(file) -> Dict:
    """Save uploaded PDF and return metadata entry (without FAISS yet)."""
    _ensure_dirs()
    raw = file.getbuffer()
    doc_id = _sha256_bytes(raw)
    safe_name = file.name.replace("/", "_").replace("\\", "_")
    pdf_path = os.path.join(PDFS_DIR, f"{doc_id}_{safe_name}")
    with open(pdf_path, "wb") as f:
        f.write(raw)
    entry = {
        "doc_id": doc_id,
        "name": safe_name,
        "size": len(raw),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pdf_path": pdf_path,
        "db_path": os.path.join(FAISS_DB_ROOT, doc_id),
        "embed_model": OLLAMA_EMBED_MODEL,
    }
    upsert_manifest_entry(entry)
    return entry


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings


def create_vector_store(db_faiss_path: str, text_chunks, ollama_model_name: str):
    os.makedirs(db_faiss_path, exist_ok=True)
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
    faiss_db.save_local(db_faiss_path)
    return faiss_db

def load_vector_store(db_faiss_path: str, ollama_model_name: str) -> Optional[FAISS]:
    try:
        return FAISS.load_local(
            db_faiss_path,
            get_embedding_model(ollama_model_name),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def retrieve_docs(faiss_db, query):
    # Default retrieval using MMR for diversity
    try:
        return faiss_db.max_marginal_relevance_search(query, k=10, fetch_k=40)
    except Exception:
        return faiss_db.similarity_search(query)

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]

def hybrid_rerank(faiss_db, query: str, top_k: int) -> List:
    """Hybrid retrieval: vector (MMR) + lexical BM25 + lightweight LLM scoring.
    Operates over vector candidates to keep latency acceptable.
    """
    candidates = retrieve_docs(faiss_db, query)
    if not candidates:
        return []
    corpus = [d.page_content for d in candidates]
    tokenized = [_tokenize(c) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    bm_scores = bm25.get_scores(_tokenize(query))

    # Lightweight LLM scoring
    def llm_score(q: str, chunk: str) -> float:
        prompt = (
            "Rate the relevance of the chunk to the query on a 0-1 scale. "
            "Reply with only a number.\n\nQuery:\n" + q + "\n\nChunk:\n" + chunk
        )
        try:
            comp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = comp.choices[0].message.content.strip()
            val = float(text)
            if val < 0:
                return 0.0
            if val > 1:
                return 1.0
            return val
        except Exception:
            return 0.5

    # Cap LLM scoring to first N for speed
    N = min(len(candidates), max(10, top_k * 3))
    llm_scores = [0.0] * len(candidates)
    for i in range(N):
        llm_scores[i] = llm_score(query, corpus[i])

    # Normalize BM25 scores
    if bm_scores:
        max_bm = max(bm_scores) or 1.0
        bm_norm = [s / max_bm for s in bm_scores]
    else:
        bm_norm = [0.0] * len(candidates)

    # Apply feedback weights if any
    def chunk_id_from_text(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()

    feedback = st.session_state.get("feedback", {})
    weights = []
    for i, text in enumerate(corpus):
        cid = chunk_id_from_text(text)
        w = 1.0
        fb = feedback.get(cid, {})
        if fb.get("pinned"):
            w += 0.2
        if fb.get("downvotes", 0) > 0:
            w -= min(0.2, 0.05 * fb.get("downvotes", 0))
        if fb.get("upvotes", 0) > 0:
            w += min(0.2, 0.05 * fb.get("upvotes", 0))
        weights.append(max(0.5, min(1.5, w)))

    # Final score: weighted blend
    final = []
    for i, d in enumerate(candidates):
        score = (0.5 * bm_norm[i] + 0.5 * llm_scores[i]) * weights[i]
        final.append((d, score, bm_norm[i], llm_scores[i], weights[i]))
    final.sort(key=lambda x: x[1], reverse=True)
    return final[:top_k]


def get_context(documents, max_chars: int = 12000):
    parts = []
    total = 0
    for doc in documents:
        text = doc.page_content
        if total + len(text) > max_chars:
            text = text[: max(0, max_chars - total)]
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts)

def summarize_history(messages: List[Dict]) -> str:
    """Create a short summary of the conversation history using Groq."""
    if not messages:
        return ""
    # Keep last ~6 exchanges for brevity
    recent = messages[-12:]
    text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent])
    prompt = f"Summarize the following conversation in 3-5 sentences focusing on the user's intent, constraints, and what has already been answered.\n\n{text}"
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception:
        return ""

def translate_text(text: str, target_lang: str) -> str:
    """Translate text using Groq; target_lang like 'en', 'fr'."""
    if not text:
        return text
    prompt = f"Translate the following text to {target_lang}. Keep legal terms precise.\n\n{text}"
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return comp.choices[0].message.content
    except Exception:
        return text


def answer_query(documents, query, memory: str = ""):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    final_prompt = prompt.format(question=query, context=context, memory=memory)
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        # If the model is decommissioned or invalid, resolve a new one and retry once.
        msg = getattr(e, "message", str(e))
        if "model" in msg and ("decommissioned" in msg or "not found" in msg):
            fallback_model = resolve_groq_model(groq_client)
            completion = groq_client.chat.completions.create(
                model=fallback_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
            )
            return completion.choices[0].message.content
        raise

def suggest_followups(question: str, answer: str) -> List[str]:
    prompt = (
        "Based on the user's question and the assistant's answer, "
        "propose three concise follow-up questions that would help clarify legal context, cite relevant articles, or refine the scope. "
        "Return them as a bullet list without numbering.\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = completion.choices[0].message.content
        lines = [l.strip("- ") for l in text.splitlines() if l.strip()]
        # Keep top 3 suggestions
        return [l for l in lines if l][:3]
    except Exception:
        return []

def extract_graph(answer: str, sources: List) -> Dict:
    """Extract a simple knowledge graph (entities, relations) from the answer and sources."""
    src_text = "\n\n".join([getattr(s, "page_content", "") for s in sources])
    schema = (
        "Return JSON with keys 'nodes' and 'edges'. 'nodes' is a list of objects {id, label, type}. "
        "'edges' is a list of objects {source, target, label}. Keep it concise and legally meaningful."
    )
    prompt = f"Build a knowledge graph from the answer and sources. {schema}\n\nAnswer:\n{answer}\n\nSources:\n{src_text}"
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = comp.choices[0].message.content
        data = json.loads(text)
        return data
    except Exception:
        return {"nodes": [], "edges": []}

def render_graph(graph_data: Dict):
    net = Network(height="500px", width="100%", notebook=False, directed=True)
    for n in graph_data.get("nodes", []):
        net.add_node(n.get("id", n.get("label", "?")), label=n.get("label", ""), title=n.get("type", ""))
    for e in graph_data.get("edges", []):
        net.add_edge(e.get("source", ""), e.get("target", ""), title=e.get("label", ""))
    html = net.generate_html("graph.html")
    st.components.v1.html(html, height=520, scrolling=True)

def _pdf_path_to_data_url(pdf_path: str) -> Optional[str]:
    try:
        with open(pdf_path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:application/pdf;base64,{b64}"
    except Exception:
        return None

def render_pdf_viewer(pdf_path: str, page: Optional[int] = None, search: Optional[str] = None, height: int = 700):
    data_url = _pdf_path_to_data_url(pdf_path)
    if not data_url:
        st.warning("Unable to load PDF source for viewer.")
        return
    file_param = urllib.parse.quote(data_url, safe="")
    hash_params = []
    if page:
        hash_params.append(f"page={int(page)}")
    if search:
        # Limit search length to avoid huge URLs
        q = urllib.parse.quote(str(search)[:200])
        hash_params.append(f"search={q}")
        hash_params.append("phrase=true")
    hash_str = ("#" + "&".join(hash_params)) if hash_params else ""
    # Use official PDF.js viewer hosted on GitHub Pages
    viewer = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={file_param}{hash_str}"
    st.components.v1.iframe(viewer, height=height)


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "selected_doc_id" not in st.session_state:
    st.session_state["selected_doc_id"] = None
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

# Sidebar: Document Library
with st.sidebar:
    st.header("Library")
    manifest = load_manifest()
    options = {f"{e.get('name')} ({e.get('doc_id')[:8]})": e.get("doc_id") for e in manifest}
    selected_label = st.selectbox("Select indexed document", list(options.keys()) or ["(none)"])
    selected_doc_id = options.get(selected_label)
    st.session_state["selected_doc_id"] = selected_doc_id

    uploaded_file_sidebar = st.file_uploader("Add PDF to library", type="pdf", accept_multiple_files=False)
    if uploaded_file_sidebar is not None:
        if st.button("Index PDF"):
            entry = upload_pdf(uploaded_file_sidebar)
            docs = load_pdf(entry["pdf_path"])
            chunks = create_chunks(docs)
            create_vector_store(entry["db_path"], chunks, entry["embed_model"])
            st.success(f"Indexed: {entry['name']}")
            manifest = load_manifest()  # refresh

    if selected_doc_id:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Delete selected index", type="secondary"):
                # Remove index directory and manifest entry
                entry = next((e for e in manifest if e.get("doc_id") == selected_doc_id), None)
                if entry and os.path.isdir(entry["db_path"]):
                    try:
                        for root, dirs, files in os.walk(entry["db_path"], topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                        os.rmdir(entry["db_path"])
                    except Exception:
                        pass
                delete_manifest_entry(selected_doc_id)
                st.session_state["selected_doc_id"] = None
                st.success("Index deleted.")
        with col_b:
            if st.button("Rebuild index"):
                entry = next((e for e in manifest if e.get("doc_id") == selected_doc_id), None)
                if entry:
                    docs = load_pdf(entry["pdf_path"])
                    chunks = create_chunks(docs)
                    create_vector_store(entry["db_path"], chunks, entry["embed_model"])
                    st.success("Index rebuilt.")


uploaded_file = st.file_uploader("Upload PDF (session-only)", type="pdf", accept_multiple_files=False)

user_query = st.text_area("Enter your prompt:", key="user_query", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if not user_query:
        st.error("Please enter a question.")
    else:
        faiss_db = None
        docs_for_answer = None

        # Priority 1: Use selected indexed document from sidebar
        selected_id = st.session_state.get("selected_doc_id")
        if selected_id:
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_id), None)
            if entry:
                faiss_db = load_vector_store(entry["db_path"], entry["embed_model"])
                if faiss_db is None:
                    # Attempt rebuild if index missing
                    source_docs = load_pdf(entry["pdf_path"])
                    chunks = create_chunks(source_docs)
                    faiss_db = create_vector_store(entry["db_path"], chunks, entry["embed_model"])
        # Priority 2: Session-only uploaded file (no persistence)
        elif uploaded_file:
            key = f"faiss::{uploaded_file.name}::{OLLAMA_EMBED_MODEL}"
            if key in st.session_state:
                faiss_db = st.session_state[key]
            else:
                temp_entry = upload_pdf(uploaded_file)  # also adds to manifest
                documents = load_pdf(temp_entry["pdf_path"])  # use persisted path to avoid temp issues
                text_chunks = create_chunks(documents)
                faiss_db = create_vector_store(temp_entry["db_path"], text_chunks, OLLAMA_EMBED_MODEL)
                st.session_state[key] = faiss_db
        else:
            st.error("Select a document from the sidebar or upload a PDF.")

        if faiss_db:
            # Multilingual: detect and translate query to English for retrieval
            try:
                q_lang = detect(user_query)
            except LangDetectException:
                q_lang = "en"
            query_for_retrieval = user_query
            if q_lang != "en":
                query_for_retrieval = translate_text(user_query, "en")

            ranked = hybrid_rerank(faiss_db, query_for_retrieval, top_k)
            retrieved_docs = [d for (d, _, _, _, _) in ranked]

            # Conversation memory
            st.session_state["messages"].append({"role": "user", "content": user_query})
            memory = summarize_history(st.session_state["messages"]) if len(st.session_state["messages"]) > 0 else ""
            response = answer_query(documents=retrieved_docs, query=query_for_retrieval, memory=memory)
            # Translate response back to original language if needed
            if q_lang != "en":
                response = translate_text(response, q_lang)
            st.session_state["messages"].append({"role": "assistant", "content": response})

            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response)

            # Follow-up suggestions
            with st.expander("Follow-up suggestions"):
                suggestions = suggest_followups(user_query, response)
                for s in suggestions:
                    if st.button(s):
                        st.session_state["user_query"] = s
                        st.experimental_rerun()

            with st.expander(f"Sources (top {top_k})"):
                for i, tup in enumerate(ranked, start=1):
                    d, score, bm_s, llm_s, w = tup
                    md = getattr(d, "metadata", {}) or {}
                    page = md.get("page", md.get("source", "unknown"))
                    preview = (d.page_content[:750] + ("…" if len(d.page_content) > 750 else ""))
                    st.markdown(f"**{i}.** Page: {page} | Score: {score:.2f} (bm={bm_s:.2f}, llm={llm_s:.2f}, w={w:.2f})\n\n{preview}")

                    # Feedback controls
                    cid = hashlib.sha256(d.page_content.encode("utf-8", errors="ignore")).hexdigest()
                    cols = st.columns(4)
                    if cols[0].button("👍", key=f"up_{cid}_{i}"):
                        fb = st.session_state["feedback"].get(cid, {"upvotes": 0, "downvotes": 0, "pinned": False})
                        fb["upvotes"] = fb.get("upvotes", 0) + 1
                        st.session_state["feedback"][cid] = fb
                        st.success("Marked helpful")
                    if cols[1].button("👎", key=f"down_{cid}_{i}"):
                        fb = st.session_state["feedback"].get(cid, {"upvotes": 0, "downvotes": 0, "pinned": False})
                        fb["downvotes"] = fb.get("downvotes", 0) + 1
                        st.session_state["feedback"][cid] = fb
                        st.info("Marked less relevant")
                    if cols[2].button("📌 Pin", key=f"pin_{cid}_{i}"):
                        fb = st.session_state["feedback"].get(cid, {"upvotes": 0, "downvotes": 0, "pinned": False})
                        fb["pinned"] = True
                        st.session_state["feedback"][cid] = fb
                        st.success("Pinned for future queries")
                    # PDF viewer
                    src_path = md.get("source") or md.get("file_path")
                    page_num = md.get("page") if isinstance(md.get("page"), int) else None
                    if src_path and cols[3].button("📄 View", key=f"view_{cid}_{i}"):
                        render_pdf_viewer(src_path, page=page_num, search=user_query if user_query else d.page_content[:80])

            # Knowledge Graph
            with st.expander("Knowledge Graph"):
                graph = extract_graph(response, retrieved_docs)
                render_graph(graph)

