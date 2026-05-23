import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --surface: #ffffff;
                --muted: #667085;
                --border: #d0d5dd;
                --accent: #0f766e;
                --accent-soft: #ccfbf1;
                --ink: #101828;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(204, 251, 241, 0.55), transparent 28rem),
                    linear-gradient(180deg, #f8fafc 0%, #eef2f6 100%);
                color: var(--ink);
            }

            .main .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 2.5rem;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--border);
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            .hero {
                border: 1px solid var(--border);
                background: rgba(255, 255, 255, 0.88);
                border-radius: 8px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            }

            .hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 2rem;
                line-height: 1.2;
            }

            .hero p {
                color: var(--muted);
                margin: 0;
                max-width: 840px;
            }

            .metric-row {
                display: grid;
                gap: 0.75rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin-bottom: 1.1rem;
            }

            .metric-card {
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--surface);
                padding: 0.85rem 1rem;
            }

            .metric-card span {
                display: block;
                color: var(--muted);
                font-size: 0.8rem;
                margin-bottom: 0.15rem;
            }

            .metric-card strong {
                color: var(--ink);
                font-size: 1rem;
            }

            .small-note {
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: 0.35rem;
            }

            .workflow {
                border: 1px solid var(--border);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.86);
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
            }

            .workflow strong {
                display: block;
                margin-bottom: 0.35rem;
            }

            .workflow p {
                color: var(--muted);
                margin: 0.25rem 0;
            }

            .stChatMessage {
                border: 1px solid rgba(208, 213, 221, 0.8);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.86);
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 8px;
                border-color: var(--border);
            }

            @media (max-width: 760px) {
                .metric-row {
                    grid-template-columns: 1fr;
                }

                .hero h1 {
                    font-size: 1.55rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1200,
        groq_api_key=api_key,
    )


@dataclass
class TfidfIndex:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    matrix: object
    chunks: list
    page_count: int
    chunk_count: int


@st.cache_resource(show_spinner=False)
def process_pdfs(file_payloads: tuple[tuple[str, bytes], ...]) -> TfidfIndex | None:
    documents = []
    temp_paths: list[str] = []

    try:
        for file_name, file_bytes in file_payloads:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_paths.append(tmp.name)

            loader = PyPDFLoader(temp_paths[-1])
            docs = loader.load()

            readable_docs = [doc for doc in docs if doc.page_content.strip()]
            if not readable_docs:
                st.warning(f"{file_name} has no readable text.")
                continue

            for doc in readable_docs:
                doc.metadata["source"] = file_name

            documents.extend(readable_docs)
    finally:
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=260)
    chunks = splitter.split_documents(documents)

    if not chunks:
        return None

    chunk_texts = [chunk.page_content for chunk in chunks]
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        max_features=90000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=45000,
    )
    word_matrix = word_vectorizer.fit_transform(chunk_texts)
    char_matrix = char_vectorizer.fit_transform(chunk_texts)
    matrix = hstack([word_matrix, char_matrix]).tocsr()

    return TfidfIndex(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        matrix=matrix,
        chunks=chunks,
        page_count=len(documents),
        chunk_count=len(chunks),
    )


def retrieve_context(index: TfidfIndex, query: str, k: int = 10):
    word_query = index.word_vectorizer.transform([query])
    char_query = index.char_vectorizer.transform([query])
    query_vector = hstack([word_query, char_query]).tocsr()
    scores = (index.matrix @ query_vector.T).toarray().ravel()
    ranked_indexes = scores.argsort()[-k:][::-1]
    selected_indexes = set()

    for chunk_index in ranked_indexes:
        if scores[chunk_index] <= 0:
            continue
        selected_indexes.add(chunk_index)
        if chunk_index > 0:
            selected_indexes.add(chunk_index - 1)
        if chunk_index + 1 < len(index.chunks):
            selected_indexes.add(chunk_index + 1)

    return [index.chunks[i] for i in sorted(selected_indexes)]


def build_search_query(user_query: str) -> str:
    words = re.findall(r"[A-Za-z]+", user_query.lower())
    generic_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "about",
        "tell",
        "me",
        "explain",
        "purpose",
        "uses",
        "use",
        "types",
        "type",
        "list",
    }
    focused_words = [word for word in words if word not in generic_words]
    return " ".join(focused_words) or user_query


def get_data_pdfs() -> list[Path]:
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.pdf"))


def build_rag_question(user_query: str) -> str:
    clean_query = user_query.strip()
    if not clean_query:
        return clean_query

    words = re.findall(r"[A-Za-z]+", clean_query)
    looks_like_topic = len(words) <= 4 and not clean_query.endswith("?")
    if looks_like_topic:
        return (
            f"Give a detailed encyclopedia-style explanation of {clean_query}. "
            "Include definition, causes, symptoms, diagnosis, treatment, prevention, "
            "and related cautions if the document provides them."
        )

    if len(words) <= 5:
        return (
            f"{clean_query}. Provide a detailed answer from the medical encyclopedia, "
            "including key facts and related details."
        )

    return clean_query


def build_lab_question(user_query: str) -> str:
    clean_query = user_query.strip()
    if not clean_query:
        return (
            "Explain this lab report in simple language. Highlight important "
            "values, abnormal high or low results if reference ranges are shown, "
            "and what the patient should discuss with a doctor."
        )

    words = re.findall(r"[A-Za-z]+", clean_query)
    if len(words) <= 5:
        return (
            f"{clean_query}. Explain using the uploaded lab report. Include the "
            "reported value, unit, reference range, whether it appears high/low/normal "
            "if available, and what it generally means."
        )

    return clean_query


def format_context(docs, max_chars: int = 14000) -> str:
    context_parts = []
    used_chars = 0

    for doc in docs:
        source = doc.metadata.get("source", "Uploaded PDF")
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        text = doc.page_content.strip()
        if not text:
            continue

        part = f"Source: {source}, {page_label}\n{text}"
        if used_chars + len(part) > max_chars:
            remaining = max_chars - used_chars
            if remaining > 500:
                context_parts.append(part[:remaining])
            break

        context_parts.append(part)
        used_chars += len(part)

    return "\n\n---\n\n".join(context_parts)


PROMPT = PromptTemplate(
    template="""You are a careful medical assistant AI.

Use only the provided context to answer the question.
If the answer is not present in the context, say:
"I couldn't find this in the documents."

Give a useful, detailed answer. Prefer 4-8 bullet points or short paragraphs
when the context supports it. Include important definitions, causes, symptoms,
treatment, prevention, and cautions when they appear in the context. If the
question is broad, summarize the major categories found in the context and say
that the document contains many more entries.

Do not diagnose. Encourage the user to consult a qualified clinician for urgent,
severe, or personal medical decisions.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


LAB_REPORT_PROMPT = PromptTemplate(
    template="""You are a careful medical report explainer for patients.

Use only the uploaded lab report context to answer. Do not invent values.
If a value, unit, or reference range is not present, say that it is not shown in
the report.

When answering, be practical and detailed:
- Start with a short plain-language summary.
- List important test values found in the report.
- If the report includes reference ranges or high/low flags, explain which
  values appear high, low, or within range.
- Explain what those tests generally relate to.
- Mention possible follow-up questions for the patient's doctor.

Do not diagnose, prescribe medicine, or claim an emergency unless the report
clearly says so. For urgent symptoms or very abnormal results, advise the
patient to contact a qualified clinician promptly.

Lab report context:
{context}

Patient question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


local_css()

st.markdown(
    """
    <section class="hero">
        <h1>Medical Chatbot</h1>
        <p>Upload medical books or patient lab reports, then ask questions grounded in the selected PDF.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        ["Medical Encyclopedia", "Lab Report", "General AI"],
        label_visibility="collapsed",
    )
    data_pdfs = get_data_pdfs()

    uploaded_files = st.file_uploader(
        "Upload medical PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        disabled=mode == "General AI",
    )

    use_data_pdfs = False
    if data_pdfs and mode in {"Medical Encyclopedia", "Lab Report"}:
        use_data_pdfs = st.checkbox(
            "Use PDFs from data folder",
            value=not uploaded_files,
        )
        with st.expander("Data folder PDFs"):
            for pdf_path in data_pdfs:
                st.write(pdf_path.name)

    if mode == "Lab Report":
        st.caption("Upload any readable patient lab report PDF and ask about its values.")
    elif mode == "Medical Encyclopedia":
        st.caption("Ask medical questions from the selected reference PDF.")
    else:
        st.caption("General AI mode does not use uploaded PDFs.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

api_key = get_secret("GROQ_API_KEY")
has_api_key = bool(api_key)
uploaded_payloads = tuple((file.name, file.getvalue()) for file in uploaded_files or [])
data_payloads = tuple((path.name, path.read_bytes()) for path in data_pdfs) if use_data_pdfs else ()
file_payloads = uploaded_payloads or data_payloads
pdf_count = len(file_payloads)
pdf_source = "Uploaded" if uploaded_payloads else "Data folder" if data_payloads else "None"
current_mode_label = mode

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card"><span>Mode</span><strong>{current_mode_label}</strong></div>
        <div class="metric-card"><span>PDFs selected</span><strong>{pdf_count} ({pdf_source})</strong></div>
        <div class="metric-card"><span>Groq key</span><strong>{"Ready" if has_api_key else "Missing"}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if mode == "Lab Report":
    st.markdown(
        """
        <div class="workflow">
            <strong>Lab report workflow</strong>
            <p>1. Select Lab Report mode.</p>
            <p>2. Upload a patient report PDF, such as CBC, blood sugar, HbA1c, lipid profile, LFT, KFT, or thyroid report.</p>
            <p>3. Ask questions like "Which values are abnormal?", "Explain this report", or "What should I ask my doctor?"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif mode == "Medical Encyclopedia":
    st.markdown(
        """
        <div class="workflow">
            <strong>Medical encyclopedia workflow</strong>
            <p>Upload or select a medical reference PDF, then ask disease, symptom, treatment, or prevention questions from that document.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not has_api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or set it as an environment variable."
    )
    st.stop()

llm = get_llm(api_key)
qa = None

if mode in {"Medical Encyclopedia", "Lab Report"}:
    if file_payloads:
        try:
            with st.spinner("Reading PDFs and building a fast local search index..."):
                index_result = process_pdfs(file_payloads)
        except Exception as exc:
            index_result = None
            st.error(f"Could not index the uploaded PDF: {exc}")

        if index_result:
            qa = index_result
            st.success(f"Indexed {qa.page_count} pages into {qa.chunk_count} searchable chunks.")
        else:
            st.warning("No readable text was found in the uploaded PDFs.")
    else:
        st.info("Upload PDFs or enable the data folder PDFs option in the sidebar.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if mode == "Lab Report":
    placeholder = "Ask about the uploaded lab report..."
elif mode == "Medical Encyclopedia":
    placeholder = "Ask about the uploaded medical PDFs..."
else:
    placeholder = "Ask a general medical question..."

query = st.chat_input(
    placeholder,
    disabled=(mode in {"Medical Encyclopedia", "Lab Report"} and qa is None),
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            if mode in {"Medical Encyclopedia", "Lab Report"}:
                rag_question = build_lab_question(query) if mode == "Lab Report" else build_rag_question(query)
                search_query = build_search_query(query)
                sources = retrieve_context(qa, search_query)
                if not sources:
                    answer = "I couldn't find this in the documents."
                    st.markdown(answer)
                else:
                    context = format_context(sources)
                    active_prompt = LAB_REPORT_PROMPT if mode == "Lab Report" else PROMPT
                    answer = llm.invoke(active_prompt.format(context=context, question=rag_question)).content
                    st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for index, doc in enumerate(sources, start=1):
                            source = doc.metadata.get("source", "Uploaded PDF")
                            page = doc.metadata.get("page")
                            page_label = f", page {page + 1}" if isinstance(page, int) else ""
                            st.write(f"{index}. {source}{page_label}")
            else:
                answer = llm.invoke(query).content
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            error_message = f"Something went wrong while generating the answer: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

st.markdown(
    '<p class="small-note">This app is for education and document lookup only. It is not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True,
)
