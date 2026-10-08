import streamlit as st
import pymupdf
import hashlib

from embeddings import embed_chunks, embed_query
from search import upload_chunks, search_chunks
from generation import generate_answer

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
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

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
    type = ["pdf", "text"]
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
        upload_results = upload_chunks(chunks, embeddings)

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
        query_embedding = embed_query(question)
        search_results = search_chunks(query_embedding)
        answer = generate_answer(question, search_results)

        st.subheader("AI Answer")
        st.write(answer)

        st.subheader("Retrieved evidence")

        for result in search_results:
            st.write(result["content"])

    st.write("Number of chunks:", len(chunks))
    st.write("Number of embeddings:", len(embeddings))
    st.write("Embedding dimensions:", len(embeddings[0]))

    st.subheader("Extracted Text Preview")
    st.text(extracted_text[:2000])
    st.write("Numbers of chunks", len(chunks))
    st.subheader("Chunk 1")
    st.text(chunks[0])
    st.subheader("Chunk 2")
    st.text(chunks[1])