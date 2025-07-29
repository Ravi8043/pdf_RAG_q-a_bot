import fitz  # PyMuPDF
from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = None
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""
    finally:
        if doc:
            doc.close()
pdf_path = "C:\\Users\\racha\\Desktop\\pdf_qa_bot\\The Art and Science of Effective AI Prompting_ Opt.pdf"



def chunk_text(
    raw_text: str,
    max_chunk_size: int = 300,
    overlap: int = 50
) -> list[str]:
    """
    Splits long text into smaller overlapping chunks.
    
    Args:
        raw_text (str): The full text extracted from the PDF.
        max_chunk_size (int): Maximum characters per chunk.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(raw_text):
        end = start + max_chunk_size
        chunk = raw_text[start:end]
        chunks.append(chunk.strip())
        start += max_chunk_size - overlap
    return chunks

raw_text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(raw_text, max_chunk_size=300, overlap=50)



# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all chunks
def embed_chunks(chunks: list[str]) -> list[np.ndarray]:
    embeddings = model.encode(chunks, show_progress_bar=True)
    return list(embeddings)


embeddings = embed_chunks(chunks)


if not embeddings or len(embeddings) == 0:
    raise ValueError("No embeddings found!")


embedding_matrix = np.asarray(embeddings, dtype=np.float32)
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

def embed_query(text: str) -> np.ndarray:
    query_vector = model.encode([text])
    query_vector = np.asarray(query_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)
    return query_vector
def search_similar_chunks(query: str, top_k: int = 3):
    query_vector = embed_query(query)
    scores, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]


load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)

def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an expert assistant analyzing a document about AI prompting techniques.

Based on the following context, provide a comprehensive and accurate answer to the question.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Use only information from the provided context
- If the context doesn't contain enough information, say so
- Provide specific examples when available
- Keep your answer clear and well-structured

ANSWER:"""
    return prompt.strip()




query = "tell me about prompt engineering"
relevant_chunks = search_similar_chunks(query, top_k=3)  
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=build_prompt(relevant_chunks, query)
)
