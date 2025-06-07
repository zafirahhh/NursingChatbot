# backend.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn
import os
import re
import torch
import time

start_time = time.time()
# --- Q&A Extraction and Embedding ---
def extract_qa_pairs(chunks):
    qa_pairs = []
    questions = []
    answers = []
    for chunk in chunks:
        # Try to match: <question line> (ends with ?) + one or more answer lines
        lines = chunk.split("\n")
        if len(lines) >= 2 and lines[0].strip().endswith("?"):
            q = lines[0].strip()
            # Join all subsequent lines as the answer (handles multi-line answers)
            a = "\n".join([l.strip() for l in lines[1:] if l.strip()])
            if a:
                qa_pairs.append((q, a))
                questions.append(q)
                answers.append(a)
                continue
        # Fallback: Q: ... A: ...
        match = re.match(r"Q:\s*(.*?)\nA:\s*(.*)", chunk, re.DOTALL)
        if match:
            q = match.group(1).strip()
            a = match.group(2).strip()
            qa_pairs.append((q, a))
            questions.append(q)
            answers.append(a)
    return qa_pairs, questions, answers

# Path to your knowledge base text file
KNOWLEDGE_PATH = os.path.join('data', 'nursing_guide.txt')

# Load the knowledge base as chunks (paragraphs separated by blank lines)
def load_chunks_from_text(text):
    chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    return chunks

def load_fine_chunks_from_text(text):
    raw_chunks = re.split(r'\n\s*\n', text)
    fine_chunks = []
    for chunk in raw_chunks:
        lines = [l.strip() for l in chunk.split('\n') if l.strip()]
        for line in lines:
            if len(line) > 20 and not line.isupper():
                fine_chunks.append(line)
    return fine_chunks

# Load chunks and embeddings once at startup
print('Loading knowledge base...')
with open(KNOWLEDGE_PATH, encoding='utf-8') as f:
    text = f.read()
chunks = load_chunks_from_text(text)
fine_chunks = load_fine_chunks_from_text(text)
print(f'Loaded {len(chunks)} chunks and {len(fine_chunks)} fine chunks in {time.time() - start_time:.2f} seconds.')

EMBEDDINGS_DIR = os.path.join('data', 'embeddings')
chunk_embeddings_path = os.path.join(EMBEDDINGS_DIR, 'chunk_embeddings.pt')
fine_chunk_embeddings_path = os.path.join(EMBEDDINGS_DIR, 'fine_chunk_embeddings.pt')
qa_embeddings_path = os.path.join(EMBEDDINGS_DIR, 'qa_embeddings.pt')

print('Loading model...')
model_load_start = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'Model loaded in {time.time() - model_load_start:.2f} seconds.')

print('Loading chunk embeddings...')
emb_start = time.time()
if os.path.exists(chunk_embeddings_path):
    chunk_embeddings = torch.load(chunk_embeddings_path, map_location='cpu')
    if chunk_embeddings.dtype == torch.float32:
        chunk_embeddings = chunk_embeddings.half()  # Convert to float16 to save memory
    print(f'Loaded chunk embeddings in {time.time() - emb_start:.2f} seconds. (dtype: {chunk_embeddings.dtype})')
else:
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True, dtype=torch.float16)
    print(f'Encoded chunk embeddings in {time.time() - emb_start:.2f} seconds. (dtype: {chunk_embeddings.dtype})')

print('Loading fine chunk embeddings...')
emb_start = time.time()
if os.path.exists(fine_chunk_embeddings_path):
    fine_chunk_embeddings = torch.load(fine_chunk_embeddings_path, map_location='cpu')
    if fine_chunk_embeddings.dtype == torch.float32:
        fine_chunk_embeddings = fine_chunk_embeddings.half()
    print(f'Loaded fine chunk embeddings in {time.time() - emb_start:.2f} seconds. (dtype: {fine_chunk_embeddings.dtype})')
else:
    fine_chunk_embeddings = model.encode(fine_chunks, convert_to_tensor=True, dtype=torch.float16)
    print(f'Encoded fine chunk embeddings in {time.time() - emb_start:.2f} seconds. (dtype: {fine_chunk_embeddings.dtype})')

qa_pairs, qa_questions, qa_answers = extract_qa_pairs(chunks)
print(f'Extracted {len(qa_questions)} QA pairs.')
emb_start = time.time()
if qa_questions and os.path.exists(qa_embeddings_path):
    qa_embeddings = torch.load(qa_embeddings_path, map_location='cpu')
    if qa_embeddings.dtype == torch.float32:
        qa_embeddings = qa_embeddings.half()
    print(f'Loaded QA embeddings in {time.time() - emb_start:.2f} seconds. (dtype: {qa_embeddings.dtype})')
elif qa_questions:
    qa_embeddings = model.encode(qa_questions, convert_to_tensor=True, dtype=torch.float16)
    print(f'Encoded QA embeddings in {time.time() - emb_start:.2f} seconds. (dtype: {qa_embeddings.dtype})')
else:
    qa_embeddings = None
    print('No QA embeddings.')

import gc
# Release unused variables to free memory
try:
    del text
    gc.collect()
except Exception:
    pass

print(f'Backend ready in {time.time() - start_time:.2f} seconds.')

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
        best_qa = max(qa_hits[0], key=lambda x: x['score'])
        if best_qa['score'] > 0.5:
            return {"answer": qa_answers[best_qa['corpus_id']]}
    # 2. Fallback to fine-grained chunk-based search
    hits = util.semantic_search(query_embedding, fine_chunk_embeddings, top_k=3)
    # Return the best scoring fine chunk above a lower threshold
    for hit in hits[0]:
        best_idx = hit['corpus_id']
        best_score = hit['score']
        if best_score > 0.4:
            return {"answer": fine_chunks[best_idx]}
    # 3. General fallback for non-medical questions or no good match
    general_response = "Sorry, I couldn't find a relevant answer. Please consult a healthcare professional for more information."
    return {"answer": general_response}

# Only run uvicorn if this file is executed directly (for local dev)
# In production (Render), gunicorn/uvicorn will use the correct port from the start command
if __name__ == "__main__":
    import os
    port = int(os.environ["PORT"])  # Only use the PORT env variable, no fallback
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
