# precompute_embeddings.py
# Run this script ONCE to precompute and save all embeddings for fast backend startup
import os
import torch
from sentence_transformers import SentenceTransformer
import re

KNOWLEDGE_PATH = os.path.join('data', 'nursing_guide.txt')
EMBEDDINGS_DIR = 'data/embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load text
with open(KNOWLEDGE_PATH, encoding='utf-8') as f:
    text = f.read()

def load_chunks_from_text(text):
    return [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]

def load_fine_chunks_from_text(text):
    raw_chunks = re.split(r'\n\s*\n', text)
    fine_chunks = []
    for chunk in raw_chunks:
        lines = [l.strip() for l in chunk.split('\n') if l.strip()]
        for line in lines:
            if len(line) > 20 and not line.isupper():
                fine_chunks.append(line)
    return fine_chunks

def extract_qa_pairs(chunks):
    qa_pairs = []
    questions = []
    answers = []
    for chunk in chunks:
        lines = chunk.split("\n")
        if len(lines) >= 2 and lines[0].strip().endswith("?"):
            q = lines[0].strip()
            a = "\n".join([l.strip() for l in lines[1:] if l.strip()])
            if a:
                qa_pairs.append((q, a))
                questions.append(q)
                answers.append(a)
                continue
        match = re.match(r"Q:\s*(.*?)\nA:\s*(.*)", chunk, re.DOTALL)
        if match:
            q = match.group(1).strip()
            a = match.group(2).strip()
            qa_pairs.append((q, a))
            questions.append(q)
            answers.append(a)
    return qa_pairs, questions, answers

chunks = load_chunks_from_text(text)
fine_chunks = load_fine_chunks_from_text(text)
qa_pairs, qa_questions, qa_answers = extract_qa_pairs(chunks)

model = SentenceTransformer('all-MiniLM-L6-v2')
print('Encoding chunks...')
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
torch.save(chunk_embeddings, os.path.join(EMBEDDINGS_DIR, 'chunk_embeddings.pt'))
print('Encoding fine chunks...')
fine_chunk_embeddings = model.encode(fine_chunks, convert_to_tensor=True)
torch.save(fine_chunk_embeddings, os.path.join(EMBEDDINGS_DIR, 'fine_chunk_embeddings.pt'))
if qa_questions:
    print('Encoding QA questions...')
    qa_embeddings = model.encode(qa_questions, convert_to_tensor=True)
    torch.save(qa_embeddings, os.path.join(EMBEDDINGS_DIR, 'qa_embeddings.pt'))
else:
    print('No QA questions found.')
print('Done! Embeddings saved to', EMBEDDINGS_DIR)
