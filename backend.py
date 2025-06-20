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
    # Add bullets and line breaks for clarity
    answer = re.sub(r"(?<=\))\s+", "\n• ", answer)
    answer = re.sub(r"(?<!\n)[•\-–]\s*", "\n• ", answer)
    answer = re.sub(r"\s{2,}", " ", answer)
    return answer.strip()

def extract_relevant_answer(question, matched_chunks):
    keywords = re.findall(r'\b\w+\b', question.lower())
    signs_keywords = {'signs', 'symptoms', 'features', 'indicators', 'criteria', 'distress'}
    best_block = ""
    max_score = 0

    for chunk in matched_chunks:
        lines = chunk.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            score = sum(1 for word in keywords if word in line_lower)
            has_signs = any(key in line_lower for key in signs_keywords)
            if has_signs:
                score += 3
            if re.match(r'^[\u2022\*-]', line.strip()):
                score += 2
            if score > max_score:
                max_score = score
                best_block = line.strip()
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith(('•', '-', '*')):
                        best_block += "\n" + lines[j].strip()
                    else:
                        break
    return best_block if best_block else matched_chunks[0]

def find_best_answer(user_query, chunks, chunk_embeddings, top_k=4):
    known = match_known_answer(user_query)
    if known:
        return known
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=min(top_k, len(chunks))).indices.tolist()
    combined_context = "\n\n".join([chunks[i] for i in top_indices])
    extracted = extract_relevant_answer(user_query, [combined_context])
    return format_answer(extracted)

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
