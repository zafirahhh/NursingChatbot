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

# Load chunks and embeddings once at startup
chunks = load_chunks(KNOWLEDGE_PATH)

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
            continue
        # Match: <question line>\n<answer line>
        lines = chunk.split("\n", 1)
        if len(lines) == 2:
            q = lines[0].strip()
            a = lines[1].strip()
            # Only treat as Q&A if question ends with ? and answer is not empty
            if q.endswith("?") and a:
                qa_pairs.append((q, a))
                questions.append(q)
                answers.append(a)
    return qa_pairs, questions, answers

# Load the embedding model and precompute embeddings
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
qa_pairs, qa_questions, qa_answers = extract_qa_pairs(chunks)
if qa_questions:
    qa_embeddings = model.encode(qa_questions, convert_to_tensor=True)
else:
    qa_embeddings = None

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

@app.post('/search')
async def search(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    # 1. Try Q&A semantic search first (medical/nursing domain)
    if qa_embeddings is not None and len(qa_questions) > 0:
        qa_hits = util.semantic_search(query_embedding, qa_embeddings, top_k=5)
        # Try to find an exact or near-exact match in the top 5
        for hit in qa_hits[0]:
            idx = hit['corpus_id']
            score = hit['score']
            # Normalize and compare questions for near-exact match
            if score > 0.5:
                user_q = request.query.strip().lower()
                kb_q = qa_questions[idx].strip().lower()
                if user_q == kb_q or user_q in kb_q or kb_q in user_q:
                    return {"answer": qa_answers[idx]}
        # If no near-exact match, return the best scoring answer above threshold
        best_qa = max(qa_hits[0], key=lambda x: x['score'])
        if best_qa['score'] > 0.5:
            return {"answer": qa_answers[best_qa['corpus_id']]}
    # 2. Fallback to chunk-based search (medical/nursing domain)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=1)
    best_idx = hits[0][0]['corpus_id']
    best_score = hits[0][0]['score']
    if best_score > 0.5:
        return {"answer": chunks[best_idx]}
    # 3. General fallback for non-medical questions or no good match
    general_response = "Sorry, I couldn't find a relevant answer. Please consult a healthcare professional for more information."
    return {"answer": general_response}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
