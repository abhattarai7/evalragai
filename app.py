import streamlit as st
import pymupdf
import hashlib
import time

from embeddings import embed_chunks, embed_query
from search import upload_chunks, search_chunks
from generation import generate_answer
from evaluation import evaluate_groundedness, evaluate_relevance, classify_hallucination_risk
from safety import analyze_safety

def extract_text_from_pdf(uploaded_file):
    pdfbytes = uploaded_file.read()
    document = pymupdf.open(
        stream = pdfbytes,  
        filetype= "pdf"
    )
    text = ""
    for page in document:
        text += page.get_text()

    document.close()

    return text

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])

            if end == len(text):
                break

            start = end - overlap

    return chunks


st.set_page_config(
    page_title="EvalRAG AI",
    page_icon="🤖",
    layout="wide"
)

st.title("EvalRAG AI")
st.subheader("RAG Evaluation & Observability Platform")

st.write(
    "Upload documents, ask questions, and evaluate "
    "AI-generated responses."
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type = ["pdf"]
)

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if uploaded_file is not None:
    st.success("Document uploaded successfully.")
    st.write("File name", uploaded_file.name)
    st.write("File type", uploaded_file.type)
    st.write("File size", uploaded_file.size, "bytes")
    pdf_bytes = uploaded_file.getvalue()    
    current_document_id = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.document_id != current_document_id:
        st.write("Processing new document...")
        extracted_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(extracted_text)
        embeddings = embed_chunks(chunks)
        upload_results = upload_chunks(chunks, embeddings, current_document_id)

        if not all(result.succeeded for result in upload_results):
            st.error("Some chunks failed to index.")
            st.stop()

        st.session_state.document_id = current_document_id
        st.session_state.extracted_text = extracted_text
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

    extracted_text = st.session_state.extracted_text
    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings

    st.success("Document indexed successfully in Azure AI Search.")

    # Add the question/retrieval code here
    st.subheader("Ask a Question")
    question = st.text_input("Ask a question about the document")

    if question:
        start_time = time.perf_counter()

        query_embedding = embed_query(question)
        search_results = search_chunks(query_embedding, current_document_id)
        answer, usage = generate_answer(question, search_results)

        safety_results = analyze_safety(answer)

        groundedness = evaluate_groundedness(question, answer, search_results)
        relevance = evaluate_relevance(question, answer)
        hallucination_risk = classify_hallucination_risk(groundedness)

        latency = time.perf_counter() - start_time

        st.subheader("AI Answer")
        st.write(answer)

        st.subheader("AI Evaluation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Groundedness", groundedness)

        with col2:
            st.metric("Relevance", relevance)

        with col3:
            st.metric("Hallucination Risk", hallucination_risk)

        st.subheader("Content Safety")

        safety_cols = st.columns(4)

        for i, result in enumerate(safety_results):
            with safety_cols[i]:
                st.metric(result.category, result.severity)

        st.subheader("Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Response Time", f"{latency:.2f} sec")

        with col2:
            if usage is not None:
                st.metric("Total Tokens", usage.total_tokens)

        if usage is not None:
            col3, col4 = st.columns(2)

            with col3:
                st.metric("Input Tokens", usage.input_tokens)

            with col4:
                st.metric("Output Tokens", usage.output_tokens)

        with st.expander("View Retrieved Evidence"):
            for i, result in enumerate(search_results, start=1):
                st.markdown(f"**Source Chunk {i}**")
                st.write(result["content"])