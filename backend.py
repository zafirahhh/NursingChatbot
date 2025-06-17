# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn
import os
import re
import torch
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nltk.download('punkt')

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Knowledge Base ===
KNOWLEDGE_PATH = os.path.join("data", "nursing_guide_cleaned.txt")
with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
    text = f.read()

def load_chunks_from_text(text, max_len=300):
    # group bullets with their heading
    paragraphs = re.split(r'\n\s*\n', text)
    grouped = []
    buffer = ""
    for para in paragraphs:
        if re.match(r'^[•*-]|^\d+\.', para.strip()):
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

# === Load Model and Embeddings ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

class QueryRequest(BaseModel):
    query: str

def extract_keywords(text):
    words = re.findall(r'\b\w{4,}\b', text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]

def filter_chunks_by_keywords_and_intent(query, chunks, keywords_map, intent_map):
    query_lower = query.lower()
    matched_keywords = []
    for concept, keywords in keywords_map.items():
        if all(k in query_lower for k in keywords):
            matched_keywords = keywords
            break

    matched_intent = []
    for intent, triggers in intent_map.items():
        if any(t in query_lower for t in triggers):
            matched_intent = triggers
            break

    filtered = []
    for i, chunk in enumerate(chunks):
        if matched_keywords and not all(k in chunk.lower() for k in matched_keywords):
            continue
        if matched_intent and not any(t in chunk.lower() for t in matched_intent):
            continue
        if any(skip in chunk.lower() for skip in ['figure', 'table', 'chapter']):
            continue
        filtered.append((i, chunk))

    return filtered if filtered else list(enumerate(chunks))

def find_best_answer(user_query, chunks, chunk_embeddings, top_k=5):
    keywords_map = {
        "urine": ["urine", "output"],
        "heart": ["heart", "rate"],
        "respiratory": ["respiratory", "rate"],
        "bp": ["blood", "pressure"],
        "fluid": ["fluid", "intake"],
        "temperature": ["temperature"],
        "neonate": ["neonate"],
        "shock": ["shock"],
        "child": ["child"],
    }

    intent_map = {
        "signs": ["signs", "symptoms", "indicators"],
        "treatment": ["first step", "treatment", "initial", "intervention", "manage", "management"],
        "vitals": ["rate", "temperature", "pressure", "value", "normal"],
    }

    filtered_pairs = filter_chunks_by_keywords_and_intent(user_query, chunks, keywords_map, intent_map)
    filtered_indices, filtered_chunks = zip(*filtered_pairs)

    query_embedding = model.encode(user_query, convert_to_tensor=True)
    reduced_embeddings = torch.stack([chunk_embeddings[i] for i in filtered_indices])
    hits = util.semantic_search(query_embedding, reduced_embeddings, top_k=top_k)[0]
    ranked = sorted(hits, key=lambda x: float(x['score']), reverse=True)
    top_chunks = [filtered_chunks[hit['corpus_id']] for hit in ranked[:3]]

    query_keywords = set(extract_keywords(user_query))
    action_verbs = ["give", "administer", "start", "treat", "use", "perform", "consider", "avoid", "manage", "monitor", "discontinue"]

    scored_chunks = []
    for chunk in top_chunks:
        chunk_lower = chunk.lower()
        if any(v in chunk_lower for v in action_verbs) and len(set(extract_keywords(chunk)) & query_keywords) >= 2:
            scored_chunks.append(chunk)

    if not scored_chunks:
        scored_chunks = [c for c in top_chunks if len(c.split()) > 5]

    if not scored_chunks:
        return top_chunks[0]  # fallback

    embeddings = model.encode(scored_chunks, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = int(torch.argmax(scores))
    # Cleanup and summarize the best sentence
    best = scored_chunks[best_idx]
    lines = best.split("\n")
    for line in lines:
        if len(line.split()) >= 4 and any(
            line.strip().lower().startswith(v) for v in [
                "give", "administer", "start", "treat", "use", "avoid", 
                "consider", "monitor", "perform", "discontinue"
            ]
        ):
            return line.strip()
    return best.split(".")[0].strip()  # fallback: return first sentence
    

# === Endpoints ===
@app.post("/ask")
async def ask_question(query: QueryRequest):
    answer = find_best_answer(query.query, chunks, chunk_embeddings)
    return {"answer": answer}

@app.post("/search")
async def search(query: QueryRequest):
    answer = find_best_answer(query.query, chunks, chunk_embeddings)
    return {"answer": answer}

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
