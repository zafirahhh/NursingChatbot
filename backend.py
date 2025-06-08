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

vital_signs_table = extract_vital_signs_table()

@app.post('/search')
async def search(request: QueryRequest):
    try:
        # 0. Improved vital signs table lookup with fuzzy/keyword matching
        if vital_signs_table:
            q = request.query.lower()
            # Map keywords to age groups
            age_keywords = {
                'neonate': 'neonate',
                'infant': 'infant',
                'toddler': 'toddler',
                'young child': 'young child',
                'older child': 'older child',
            }
            matched_row = None
            for keyword, label in age_keywords.items():
                if keyword in q:
                    for row in vital_signs_table:
                        if label in row['Age Group'].lower():
                            matched_row = row
                            break
                if matched_row:
                    break
            if matched_row:
                # Check for specific vital sign in query
                if 'heart rate' in q:
                    answer = f"{matched_row['Age Group']} Heart Rate: {matched_row['Heart Rate (beats/min)']}"
                elif 'respiratory rate' in q or 'resp rate' in q:
                    answer = f"{matched_row['Age Group']} Respiratory Rate: {matched_row['Respiratory Rate (breaths/min)']}"
                elif 'systolic' in q or 'bp' in q or 'blood pressure' in q:
                    answer = f"{matched_row['Age Group']} Systolic BP: {matched_row['Systolic BP (mmHg)']}"
                else:
                    answer = (f"Age Group: {matched_row['Age Group']}\n"
                              f"Heart Rate: {matched_row['Heart Rate (beats/min)']}\n"
                              f"Respiratory Rate: {matched_row['Respiratory Rate (breaths/min)']}\n"
                              f"Systolic BP: {matched_row['Systolic BP (mmHg)']}")
                return {"answer": answer}
        query_embedding = model.encode(request.query, convert_to_tensor=True, dtype=torch.float32)
        # 1. Try Q&A semantic search first (medical/nursing domain)
        if qa_embeddings is not None and len(qa_questions) > 0:
            qa_hits = util.semantic_search(query_embedding, qa_embeddings, top_k=5)
            best_qa = max(qa_hits[0], key=lambda x: x['score'])
            if best_qa['score'] > 0.5:
                return {"answer": qa_answers[best_qa['corpus_id']]}
        # 2. Fallback to fine-grained chunk-based search
        hits = util.semantic_search(query_embedding, fine_chunk_embeddings, top_k=3)
        # Return the parent chunk containing the best scoring fine chunk above a lower threshold
        for hit in hits[0]:
            best_idx = hit['corpus_id']
            best_score = hit['score']
            print(f"[DEBUG] Fine chunk idx: {best_idx}, score: {best_score}")
            if best_score > 0.4:
                fine_text = fine_chunks[best_idx]
                parent_chunk = next((chunk for chunk in chunks if fine_text in chunk), fine_text)
                print(f"[DEBUG] Parent chunk: {parent_chunk}")
                sentences = sent_tokenize(parent_chunk)
                print(f"[DEBUG] Sentences: {sentences}")
                sent_embeddings = model.encode(sentences, convert_to_tensor=True, dtype=torch.float32)
                sent_scores = util.pytorch_cos_sim(query_embedding, sent_embeddings)[0]
                print(f"[DEBUG] Sentence scores: {sent_scores}")
                best_sent_idx = int(torch.argmax(sent_scores))
                best_sentence = sentences[best_sent_idx]
                print(f"[DEBUG] Best sentence: {best_sentence}")
                # If the best sentence looks like a table header, return the next 8 lines as well
                if ("table" in best_sentence.lower() or "table of" in best_sentence.lower()) and len(sentences) > best_sent_idx + 1:
                    # Skip the header and return the next 8 lines (or as many as available)
                    answer = '\n'.join(sentences[best_sent_idx+1:best_sent_idx+9])
                    # If the answer is empty, fallback to the next best sentence
                    if not answer.strip() and len(sentences) > best_sent_idx + 1:
                        answer = sentences[best_sent_idx+1]
                elif len(best_sentence) < 30 and len(sentences) > 1:
                    top2_idx = torch.topk(sent_scores, 2).indices.tolist()
                    answer = ' '.join([sentences[i] for i in top2_idx])
                else:
                    answer = best_sentence
                print(f"[DEBUG] Final answer: {answer}")
                return {"answer": answer}
        general_response = "Sorry, I couldn't find a relevant answer. Please consult a healthcare professional for more information."
        print(f"[DEBUG] General fallback triggered.")
        return {"answer": general_response}
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
