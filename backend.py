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
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

start_time = time.time()
# --- Q&A Extraction and Embedding ---
def extract_qa_pairs(chunks):
    qa_pairs = []
    questions = []
    answers = []
    for chunk in chunks:
        # Improved: Find the first line ending with '?' as the question, rest as answer
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        q_idx = None
        for idx, line in enumerate(lines):
            if line.endswith('?'):
                q_idx = idx
                break
        if q_idx is not None and q_idx < len(lines) - 1:
            q = lines[q_idx]
            a = "\n".join(lines[q_idx + 1:]).strip()
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
# Use a smaller, public model to reduce memory usage
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print(f'Model loaded in {time.time() - model_load_start:.2f} seconds.')

def safe_load_embeddings(path, encode_fn, items, name):
    try:
        if os.path.exists(path):
            emb = torch.load(path, map_location='cpu')
            if emb.dtype != torch.float32:
                emb = emb.float()
            print(f'Loaded {name} embeddings from disk. (dtype: {emb.dtype})')
            return emb
        else:
            emb = encode_fn(items, convert_to_tensor=True, dtype=torch.float32)
            torch.save(emb, path)
            print(f'Encoded and saved {name} embeddings. (dtype: {emb.dtype})')
            return emb
    except Exception as e:
        print(f'Error loading {name} embeddings: {e}. Regenerating...')
        emb = encode_fn(items, convert_to_tensor=True, dtype=torch.float32)
        torch.save(emb, path)
        print(f'Regenerated and saved {name} embeddings. (dtype: {emb.dtype})')
        return emb

print('Loading chunk embeddings...')
chunk_embeddings = safe_load_embeddings(chunk_embeddings_path, model.encode, chunks, 'chunk')

print('Loading fine chunk embeddings...')
fine_chunk_embeddings = safe_load_embeddings(fine_chunk_embeddings_path, model.encode, fine_chunks, 'fine chunk')

qa_pairs, qa_questions, qa_answers = extract_qa_pairs(chunks)
print(f'Extracted {len(qa_questions)} QA pairs.')
if qa_questions:
    qa_embeddings = safe_load_embeddings(qa_embeddings_path, model.encode, qa_questions, 'QA')
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
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://zafirahhh.github.io"
    ],  # Explicitly allow local and GitHub Pages origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# --- Structured Table Extraction ---
def extract_vital_signs_table():
    table_path = os.path.join('data', 'nursing_guide.txt')
    with open(table_path, encoding='utf-8') as f:
        lines = f.readlines()
    table = []
    header = None
    for line in lines:
        if line.strip().startswith('# Age Group|'):
            header = [h.strip('# ').strip() for h in line.strip().split('|')]
            continue
        if header and '|' in line:
            row = [col.strip() for col in line.strip().split('|')]
            if len(row) == len(header):
                table.append(dict(zip(header, row)))
        elif header and not line.strip():
            break  # Stop at first blank line after table
    return table

def extract_all_pipe_tables():
    table_path = os.path.join('data', 'nursing_guide.txt')
    with open(table_path, encoding='utf-8') as f:
        lines = f.readlines()
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('#') and '|' in line:
            header = [h.strip('# ').strip() for h in line.strip().split('|')]
            table_rows = []
            i += 1
            while i < len(lines) and '|' in lines[i]:
                row = [col.strip() for col in lines[i].strip().split('|')]
                if len(row) == len(header):
                    table_rows.append(dict(zip(header, row)))
                i += 1
            tables.append({'header': header, 'rows': table_rows, 'title': line.strip('#').strip()})
        else:
            i += 1
    return tables

vital_signs_table = extract_vital_signs_table()
all_tables = extract_all_pipe_tables()

@app.post('/search')
async def search(request: QueryRequest):
    try:
        q = request.query.lower()
        # Table selection logic
        table_map = {
            'vital': 'age group',
            'heart rate': 'age group',
            'respiratory rate': 'age group',
            'bp': 'age group',
            'blood pressure': 'age group',
            'systolic': 'age group',
            'urine': 'normal urine output',
            'fluid': 'fluids calculator',
            'holliday': 'fluids calculator',
            'output': 'normal urine output',
            'expected systolic': 'expected systolic blood pressure',
        }
        # Find the best matching table
        selected_table = None
        for k, v in table_map.items():
            if k in q:
                for table in all_tables:
                    if v.lower() in table['title'].lower():
                        selected_table = table
                        break
            if selected_table:
                break
        if not selected_table:
            return {"answer": "Sorry, I couldn't find relevant data for your query."}
        # Try to match a row based on age/weight/description
        for row in selected_table['rows']:
            for value in row.values():
                if value.lower() in q:
                    return {"answer": str(row)}
        # If no row match, return the whole table as a string
        table_str = '\n'.join([' | '.join(selected_table['header'])] + [' | '.join(row.values()) for row in selected_table['rows']])
        return {"answer": table_str}
    except Exception as e:
        import traceback
        print(f"Error in /search: {e}")
        traceback.print_exc()
        return {"answer": "Internal server error. Please check backend logs."}

@app.get("/")
def root():
    return {"message": "Nursing Chatbot backend is running. Use the /search endpoint with POST requests."}

# Only run uvicorn if this file is executed directly (for local dev)
# In production (Render), gunicorn/uvicorn will use the correct port from the start command
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Use 8000 as default if PORT is not set
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
