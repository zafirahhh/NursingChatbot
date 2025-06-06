# backend.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn
import os
import re

# Path to your knowledge base text file
KNOWLEDGE_PATH = os.path.join('data', 'nursing_guide.txt')

# Load the knowledge base as chunks (paragraphs separated by blank lines)
def load_chunks(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    # Split on two or more newlines (paragraphs)
    chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    return chunks

chunks = load_chunks(KNOWLEDGE_PATH)

# Load the embedding model
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# --- Q&A Extraction and Embedding ---
def extract_qa_pairs(chunks):
    qa_pairs = []
    questions = []
    answers = []
    for chunk in chunks:
        # Match Q: ... A: ...
        match = re.match(r"Q:\s*(.*?)\nA:\s*(.*)", chunk, re.DOTALL)
        if match:
            q = match.group(1).strip()
            a = match.group(2).strip()
            qa_pairs.append((q, a))
            questions.append(q)
            answers.append(a)
    return qa_pairs, questions, answers

qa_pairs, qa_questions, qa_answers = extract_qa_pairs(chunks)
if qa_questions:
    qa_embeddings = model.encode(qa_questions, convert_to_tensor=True)
else:
    qa_embeddings = None

@app.post('/search')
async def search(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    # 1. Try Q&A semantic search first (medical/nursing domain)
    if qa_embeddings is not None and len(qa_questions) > 0:
        qa_hits = util.semantic_search(query_embedding, qa_embeddings, top_k=1)
        qa_best_idx = qa_hits[0][0]['corpus_id']
        qa_best_score = qa_hits[0][0]['score']
        if qa_best_score > 0.6:
            return {"answer": qa_answers[qa_best_idx]}
    # 2. Fallback to chunk-based search (medical/nursing domain)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=1)
    best_idx = hits[0][0]['corpus_id']
    best_score = hits[0][0]['score']
    if best_score > 0.4:
        return {"answer": chunks[best_idx]}
    # 3. General fallback for non-medical questions
    general_response = "I'm designed to answer nursing and medical questions. For other topics, please consult a general knowledge resource."
    return {"answer": general_response}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
