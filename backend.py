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
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk + para) < max_len:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = load_chunks_from_text(text)

# === Load Model and Chunk Embeddings ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

# === Request Schema ===
class QueryRequest(BaseModel):
    query: str

# === Filtering Logic ===
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
        filtered.append((i, chunk))

    return filtered if filtered else list(enumerate(chunks))

# === Semantic Answer Logic (shortened version) ===
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
    for hit in ranked:
        if float(hit['score']) > 0.4:
            chunk = filtered_chunks[hit['corpus_id']]
            sentences = sent_tokenize(chunk)
            if sentences:
                sent_embeddings = model.encode(sentences, convert_to_tensor=True)
                sent_scores = util.cos_sim(query_embedding, sent_embeddings)[0]
                best_sent_idx = int(torch.argmax(sent_scores))
                return sentences[best_sent_idx]
            else:
                return chunk
    return "Sorry, I could not find relevant information for that question."

# === Both Endpoints ===
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
