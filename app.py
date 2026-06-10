import streamlit as st
import pdfplumber

from utils.chunking import chunk_text
from utils.embeddings import (
    create_embeddings,
    create_query_embedding
)
from utils.vector_store import create_vector_store
from utils.retrieval import search_chunks
from utils.llm import (
    generate_answer,
    summarize_document,
    financial_analysis
)

st.set_page_config(
    page_title="Financial Document Chatbot"
)

st.title("📊 Financial Document Chatbot")

# Session State
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader(
    "Upload Financial Report",
    type=["pdf"]
)

# Detect New Upload
if uploaded_file:

    if (
        "current_file" not in st.session_state
        or st.session_state.current_file != uploaded_file.name
    ):
        st.session_state.processed = False
        st.session_state.current_file = uploaded_file.name

# Process PDF Once
if uploaded_file and not st.session_state.processed:

    with st.spinner("Processing document..."):

        text = ""

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        chunks = chunk_text(text)

        embeddings = create_embeddings(chunks)

        vector_store = create_vector_store(embeddings)

        st.session_state.text = text
        st.session_state.chunks = chunks
        st.session_state.vector_store = vector_store

        st.session_state.processed = True

# Main Interface
if st.session_state.processed:

    text = st.session_state.text
    chunks = st.session_state.chunks
    vector_store = st.session_state.vector_store

    st.success("✅ Document Ready")

    # Summary Button
    if st.button("📄 Summarize Document"):

        with st.spinner("Generating Summary..."):

            summary = summarize_document(text)

        st.subheader("Document Summary")

        st.write(summary)

    # Financial Analysis Button
    if st.button("📈 Financial Analysis"):

        with st.spinner("Analyzing Financial Report..."):

            analysis = financial_analysis(text)

        st.subheader("Financial Analysis")

        st.write(analysis)

    # Question Answering
    query = st.text_input(
        "Ask a question about the document"
    )

    if query:

        query_embedding = create_query_embedding(query)

        results = search_chunks(
            query_embedding,
            vector_store,
            chunks,
            k=3
        )

        context = "\n\n".join(results)

        with st.spinner("Generating Answer..."):

            answer = generate_answer(
                query,
                context
            )

        st.markdown("### Answer")

        st.write(answer)