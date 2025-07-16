import streamlit as st
import fitz  # PyMuPDF
import re

def clean_text(text):
    lines = text.split("\n")
    
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove lines that ONLY contain digits (e.g., "6")
        if not (stripped.isdigit() and len(stripped) < 4):
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()


st.set_page_config(page_title="AskMyPaper", layout="wide")
st.title("📄 AskMyPaper – AI Research Assistant")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    # Read and extract text
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in pdf_doc:
        full_text += clean_text(page.get_text()) + "\n"


    #st.subheader("📑 Extracted Text")
    #st.text_area("Full Text from PDF", full_text, height=300)


    from rag_pipeline import chunk_text

    chunks = chunk_text(full_text)
    #st.subheader("🧩 First Chunk")
    #st.write(chunks[0])
    st.write(f"Total Chunks: {len(chunks)}")

    from rag_pipeline import embed_chunks, retrieve_relevant_chunks, generate_answer

    # Step 1: Create embeddings and FAISS index
    index, embeddings, model = embed_chunks(chunks)

    # Step 2: User Q&A Input
    st.subheader("❓ Ask a Question")
    user_question = st.text_input("What do you want to know about this paper?")

    top_k = st.slider("🔍 Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    if user_question:
        top_chunks = retrieve_relevant_chunks(user_question, chunks, model, index, top_k=top_k)

        st.subheader("🔍 Top Matching Chunks")
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Match {i}:**")
            st.write(chunk)
            st.markdown("---")

        st.subheader("🧠 Answer from Mistral")
        answer = generate_answer(user_question, top_chunks)
        st.success(answer)

       

