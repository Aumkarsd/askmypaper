import textwrap

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks for embedding.

    Args:
        text (str): Cleaned full text from PDF.
        chunk_size (int): Approx words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List of string chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Overlap helps preserve context

    return chunks

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def embed_chunks(chunks, model_name="multi-qa-MiniLM-L6-cos-v1"):
    # Load embedding model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings, model

def retrieve_relevant_chunks(question, chunks, model, index, top_k=3):
    """
    Embeds the user question and retrieves top_k most relevant chunks from the index.
    """
    # Embed the question
    question_embedding = model.encode([question])
    
    # Search FAISS index
    distances, indices = index.search(question_embedding, top_k)
    
    # Get the top matching chunks
    matched_chunks = [chunks[i] for i in indices[0]]
    
    return matched_chunks

from langchain.llms import Ollama

def generate_answer(query, context_chunks):
    # Join the chunks as context
    context = "\n\n".join(context_chunks)

    # Create the prompt
    prompt = f"""You are an AI research assistant. Use the following context to answer the user's question accurately and concisely.

Context:
{context}

Question: {query}
Answer:"""

    # Call the Mistral model via Ollama
    llm = Ollama(model="mistral")
    return llm(prompt)

