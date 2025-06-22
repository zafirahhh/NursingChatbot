# backend.py
from fastapi import FastAPI, Request
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
import random
from typing import List

nltk.download('punkt')

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Main QA Logic
def answer_from_knowledge_base(question: str, return_summary=True):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_chunk = chunks[best_idx]

    question_keywords = set(re.findall(r'\w+', question.lower()))
    candidate_sents = [
        s.strip() for s in sent_tokenize(best_chunk)
        if 6 <= len(s.split()) <= 35 and not any(x in s for x in ['|', ':'])
    ]
    if not candidate_sents:
        return best_chunk

    for sent in candidate_sents:
        words = set(re.findall(r'\w+', sent.lower()))
        if len(words & question_keywords) >= 2:
            return sent

    comma_sents = [s for s in candidate_sents if ',' in s]
    if comma_sents:
        return max(comma_sents, key=lambda s: len(s))

    best_score = 0
    best_sent = candidate_sents[0]
    for sent in candidate_sents:
        sim = SequenceMatcher(None, question.lower(), sent.lower()).ratio()
        if sim > best_score:
            best_score = sim
            best_sent = sent

    return best_sent


# Quiz Generator
def generate_quiz_from_guide(prompt: str):
    # Example basic quiz generation logic
    return (
        "\U0001F9E0 Quiz Time!\n"
        "Q: What is the initial management for a febrile seizure?\n"
        "A. Administer antibiotics\n"
        "B. Cool the child\n"
        "C. Administer paracetamol\n"
        "D. Call ICU\n"
        "\nType the correct option (A/B/C/D)."
    )

# === Request Schema ===
class QueryRequest(BaseModel):
    query: str

class QuizAnswer(BaseModel):
    question: str
    answer: str

class QuizEvaluationRequest(BaseModel):
    responses: List[QuizAnswer]

# === Helper Functions ===
def clean_paragraph(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"^[\u2022\-\u2013\*\d\.\s]+", "", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    paragraph = " ".join(cleaned_lines)
    paragraph = re.sub(r'\s+', ' ', paragraph).strip()
    return paragraph

def extract_summary_sentences(text: str, max_sentences=3) -> str:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 5]
    formula_lines = [
        line for line in lines
        if re.search(r'70\s*\+\s*\(?.*?age.*?\)?[^\|\n]*', line, re.IGNORECASE)
        or re.search(r'expected systolic bp.*?70.*?age', line, re.IGNORECASE)
    ]
    if formula_lines:
        matches = []
        for line in formula_lines:
            found = re.findall(r'70\s*\+\s*\(?.*?age.*?\)?[^\|\n]*', line, re.IGNORECASE)
            matches.extend(found)
        if matches:
            return '\n'.join(f'- Expected systolic BP formula: {m.strip()}' for m in matches[:max_sentences])
    sentences = [s.strip() for s in sent_tokenize(text) if 10 < len(s.strip()) < 200]
    key_sents = [s for s in sentences if ':' not in s and '|' not in s and len(s.split()) <= 25]
    fallback = key_sents[:max_sentences] or sentences[:max_sentences]
    return '\n'.join(f'- {s}' for s in fallback) if fallback else 'No relevant sentence found.'

def find_best_answer(user_query, chunks, chunk_embeddings, top_k=2):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=min(top_k, len(chunks))).indices.tolist()

    best_chunk = chunks[top_indices[0]]
    summary = answer_from_knowledge_base(user_query, return_summary=True)

    return {
        "summary": summary,
        "full": clean_paragraph(best_chunk)
    }


@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    session = data.get("session", "general")

    if session == "quiz":
        return {"answer": generate_quiz_from_guide(question)}
    else:
        result = find_best_answer(question, chunks, chunk_embeddings)
        return result

@app.post("/search")
async def search(query: QueryRequest):
    result = find_best_answer(query.query, chunks, chunk_embeddings)
    return result

@app.get("/quiz")
def generate_quiz(n: int = 5, topic: str = None):
    filtered_chunks = [c for c in chunks if topic and topic.lower() in c.lower()] if topic else chunks
    quiz = []
    attempts = 0
    max_attempts = n * 10  # Prevent infinite loop
    while len(quiz) < n and attempts < max_attempts:
        chunk = random.choice(filtered_chunks)
        sentences = [s.strip() for s in sent_tokenize(chunk) if 6 <= len(s.split()) <= 25]
        sentences = list(set(sentences))  # Deduplicate
        if len(sentences) < 5:
            attempts += 1
            continue  # Ensure at least 5 unique options
        correct = random.choice(sentences)
        distractors = [s for s in sentences if s != correct]
        selected = random.sample(distractors, 4) + [correct]
        random.shuffle(selected)
        quiz.append({
            "question": ("What is the correct statement regarding " + topic) if topic else "What is the correct statement regarding this medical topic?",
            "options": selected,
            "answer": correct,
            "context": chunk
        })
        attempts += 1
    return {"quiz": quiz}

@app.post("/quiz/evaluate")
def evaluate_quiz(request: QuizEvaluationRequest):
    feedback = []
    for response in request.responses:
        question_text = response.question.lower().strip()
        given_answer = response.answer.strip()

        best_chunk = ""
        best_score = 0
        best_match = ""

        for chunk in chunks:
            for sent in sent_tokenize(chunk):
                score = SequenceMatcher(None, sent.lower(), given_answer.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = sent
                    best_chunk = chunk

        correct = best_match if best_score > 0.7 else "Unknown"
        explanation = extract_summary_sentences(best_chunk, max_sentences=2) if best_chunk else "No explanation found."

        feedback.append({
            "question": response.question,
            "your_answer": given_answer,
            "correctAnswer": correct,
            "correct": given_answer == correct,
            "explanation": explanation
        })

    return feedback

if __name__ == "__main__":
    import time
    time.sleep(2)
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
