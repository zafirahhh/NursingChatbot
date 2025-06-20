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
    best_block = ""
    max_score = 0

    for chunk in matched_chunks:
        lines = chunk.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()

            # Skip table headers, metadata
            if re.search(r'(table \d+|adapted from|figure \d+|source[:\s])', line_lower):
                continue

            score = sum(1 for word in keywords if word in line_lower)

            # Strong boost for bullet-style answers with signs/symptoms/indicators
            if re.match(r'^[\u2022\*-]', line.strip()):
                score += 3
            if re.search(r'signs|symptoms|indicated|contraindicated|criteria|manifestation|features', line_lower):
                score += 2
            if re.search(r'\d+\s*(mg|ml|mmol|bpm|hr|min)', line_lower):
                score += 1

            if score > max_score:
                max_score = score
                best_block = line.strip()

                # Extend answer if part of a list
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith(('•', '-', '*')):
                        best_block += "\n" + lines[j].strip()
                    else:
                        break

    return best_block or "Sorry, I couldn't find a clear answer in the document."

# === Semantic Answer Logic ===
def find_best_answer(user_query, chunks, chunk_embeddings, top_k=5):
    known = match_known_answer(user_query)
    if known:
        return known

    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=min(top_k, len(chunks))).indices
    matched_chunks = [chunks[i] for i in top_indices]

    return extract_relevant_answer(user_query, matched_chunks)

def smart_summarize(text, max_words=60):
    import re
    text = re.sub(r"\n+", " ", text.strip())  # Remove line breaks
    sentences = re.split(r"(?<=[.?!]) +", text)
    result = []
    word_count = 0
    for s in sentences:
        s_words = len(s.split())
        if word_count + s_words > max_words:
            break
        result.append(s)
        word_count += s_words
    return " ".join(result)

# === Endpoints ===
@app.post("/ask")
async def ask_question(query: QueryRequest):
    answer = find_best_answer(query.query, chunks, chunk_embeddings)
    clean_answer = smart_summarize(answer)
    return {"answer": clean_answer}

@app.post("/search")
async def search(query: QueryRequest):
    answer = find_best_answer(query.query, chunks, chunk_embeddings)
    clean_answer = smart_summarize(answer)
    return {"answer": clean_answer}

# === Run the Server ===
if __name__ == "__main__":
    import time
    time.sleep(2)
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
