# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn
import os
import re
import torch
import json
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from difflib import SequenceMatcher

nltk.download('punkt')

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Q&A Reference if available ===
qa_reference_path = os.path.join("data", "qa_reference.json")
qa_reference = []
if os.path.exists(qa_reference_path):
    with open(qa_reference_path, encoding="utf-8") as f:
        qa_reference = json.load(f)

def match_known_answer(query):
    best_match = None
    best_score = 0
    for pair in qa_reference:
        score = SequenceMatcher(None, query.lower(), pair["q"].lower()).ratio()
        if score > best_score:
            best_match = pair
            best_score = score
    if best_score >= 0.9:
        return best_match["a"]
    return None

# === Load Knowledge Base ===
KNOWLEDGE_PATH = os.path.join("data", "nursing_guide_cleaned.txt")
with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
    text = f.read()

def load_chunks_from_text(text, max_len=300):
    paragraphs = re.split(r'\n\s*\n', text)
    grouped = []
    buffer = ""
    for para in paragraphs:
        if re.match(r'^[\u2022\*-]|^\d+\.', para.strip()):
            buffer += para.strip() + " "
        else:
            if buffer:
                grouped.append(buffer.strip())
                buffer = ""
            grouped.append(para.strip())
    if buffer:
        grouped.append(buffer.strip())
    return [p for p in grouped if len(p.split()) > 5]

chunks = load_chunks_from_text(text)

# === Load Model and Chunk Embeddings ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

# === Request Schema ===
class QueryRequest(BaseModel):
    query: str

# === Helper Functions ===
def format_answer(answer: str) -> str:
    import re

    # Step 1: Break into lines and remove bullet symbols
    lines = answer.splitlines()
    cleaned_lines = []

    for line in lines:
        # Strip bullets like •, -, *, and whitespace
        cleaned = re.sub(r"^[\u2022\-–*•\d\.\s]+", "", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)

    # Step 2: Combine into full paragraph and fix spacing
    paragraph = " ".join(cleaned_lines)
    paragraph = re.sub(r'\s+', ' ', paragraph)
    paragraph = re.sub(r'\s*\.\s*', '. ', paragraph).strip()

    # Step 3: Capitalize first letter if missing
    if paragraph and not paragraph[0].isupper():
        paragraph = paragraph[0].upper() + paragraph[1:]

    return paragraph

def find_best_answer(user_query, chunks, chunk_embeddings, top_k=2):
    known = match_known_answer(user_query)
    if known:
        return known

    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=min(top_k, len(chunks))).indices.tolist()

    combined_context = "\n\n".join([chunks[i] for i in top_indices])
    return format_answer(combined_context)

# === Endpoints ===
@app.post("/ask")
async def ask_question(query: QueryRequest):
    answer = find_best_answer(query.query, chunks, chunk_embeddings)
    return {"answer": answer}

@app.post("/search")
async def search(query: QueryRequest):
    answer = find_best_answer(query.query, chunks, chunk_embeddings)
    return {"answer": answer}

# === Run the Server ===
if __name__ == "__main__":
    import time
    time.sleep(2)
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
