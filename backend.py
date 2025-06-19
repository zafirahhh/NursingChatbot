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
def extract_keywords(text):
    words = re.findall(r'\b\w{4,}\b', text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]

def extract_relevant_answer(question, matched_chunks):
    keywords = re.findall(r'\b\w+\b', question.lower())
    best_line = None
    max_score = 0

    for chunk in matched_chunks:
        lines = sent_tokenize(chunk)
        for line in lines:
            line_lower = line.lower()

            # Skip generic headings, table references, source lines
            if re.search(r'table\s*\d|figure\s*\d|adapted from|source[:\s]', line_lower):
                continue

            score = sum(1 for word in keywords if word in line_lower)

            # Boost score if numeric/clinical actionable info
            if re.search(r'\b\d+\s*(mg|g|ml|mmol|hours?|mins?)\b', line_lower):
                score += 2
            if re.search(r'double|increase|reduce|adjust|infusion|bolus|dialysis|resuscitation|dose|indicated|symptom|signs|history|criteria', line_lower):
                score += 1

            if score > max_score:
                max_score = score
                best_line = line.strip()

    return best_line or "Sorry, I couldn't find a clear answer in the document."

# === Semantic Answer Logic ===
def find_best_answer(user_query, chunks, chunk_embeddings, top_k=5):
    # Try known Q&A match first
    known = match_known_answer(user_query)
    if known:
        return known

    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=min(top_k, len(chunks))).indices
    matched_chunks = [chunks[i] for i in top_indices]

    return extract_relevant_answer(user_query, matched_chunks)

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
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
