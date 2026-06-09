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
    summarize_document
)

st.set_page_config(
    page_title="Financial Document Chatbot"
)

st.title("Financial Document Chatbot")

uploaded_file = st.file_uploader(
    "Upload Financial Report",
    type=["pdf"]
)

if uploaded_file:

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

    st.success("Document Ready")

    # Summary Button
    if st.button("Summarize Document"):

        with st.spinner("Generating Summary..."):

            summary = summarize_document(text)

        st.subheader("Document Summary")

        st.write(summary)

    # Question Section
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