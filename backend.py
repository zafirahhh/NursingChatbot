# backend.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn
import os

# Path to your knowledge base text file
KNOWLEDGE_PATH = os.path.join('data', 'nursing_guide.txt')

# Load the knowledge base
with open(KNOWLEDGE_PATH, encoding='utf-8') as f:
    knowledge_lines = [line.strip() for line in f if line.strip()]

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_embeddings = model.encode(knowledge_lines, convert_to_tensor=True)

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
    hits = util.semantic_search(query_embedding, knowledge_embeddings, top_k=1)
    best_idx = hits[0][0]['corpus_id']
    best_score = hits[0][0]['score']
    answer = knowledge_lines[best_idx] if best_score > 0.4 else "No relevant information found."
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
