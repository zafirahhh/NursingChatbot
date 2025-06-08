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
def extract_all_pipe_tables():
    table_path = os.path.join('data', 'nursing_guide.txt')
    with open(table_path, encoding='utf-8') as f:
        lines = f.readlines()
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('#') and '|' in line:
            header = [h.strip('# ').strip() for h in line.strip().split('|')]
            table_rows = []
            i += 1
            while i < len(lines) and '|' in lines[i]:
                row = [col.strip() for col in lines[i].strip().split('|')]
                if len(row) == len(header):
                    table_rows.append(dict(zip(header, row)))
                i += 1
            tables.append({'header': header, 'rows': table_rows, 'title': line.strip('#').strip()})
        else:
            i += 1
    return tables

all_tables = extract_all_pipe_tables()

# --- Table Row Embedding Preparation ---
table_row_texts = []  # List of (table_title, row_dict, row_text)
table_row_lookup = []
for table in all_tables:
    for row in table['rows']:
        # Join all values for semantic search, include table title and column names for context
        row_text = f"Table: {table['title']} | " + ' | '.join([f"{col}: {row[col]}" for col in table['header']])
        table_row_texts.append(row_text)
        table_row_lookup.append((table['title'], row))

# Precompute or load embeddings for all table rows
TABLE_ROW_EMB_PATH = os.path.join('data', 'embeddings', 'table_row_embeddings.pt')
def safe_load_table_row_embeddings():
    try:
        if os.path.exists(TABLE_ROW_EMB_PATH):
            emb = torch.load(TABLE_ROW_EMB_PATH, map_location='cpu')
            if emb.dtype != torch.float32:
                emb = emb.float()
            print(f'Loaded table row embeddings from disk. (dtype: {emb.dtype})')
            return emb
        else:
            emb = model.encode(table_row_texts, convert_to_tensor=True, dtype=torch.float32)
            torch.save(emb, TABLE_ROW_EMB_PATH)
            print(f'Encoded and saved table row embeddings. (dtype: {emb.dtype})')
            return emb
    except Exception as e:
        print(f'Error loading table row embeddings: {e}. Regenerating...')
        emb = model.encode(table_row_texts, convert_to_tensor=True, dtype=torch.float32)
        torch.save(emb, TABLE_ROW_EMB_PATH)
        print(f'Regenerated and saved table row embeddings. (dtype: {emb.dtype})')
        return emb

table_row_embeddings = safe_load_table_row_embeddings()

# --- Table Cell Embedding Preparation ---
table_cell_texts = []  # List of (table_title, row_label, column_name, value, full_context)
table_cell_lookup = []
for table in all_tables:
    for row in table['rows']:
        row_label = row.get(table['header'][0], '')  # Assume first column is the row label (e.g., Age Group)
        for col in table['header']:
            value = row[col]
            # Compose a context string for semantic search
            cell_text = f"Table: {table['title']} | Row: {row_label} | Column: {col} | Value: {value}"
            table_cell_texts.append(cell_text)
            table_cell_lookup.append((table['title'], row_label, col, value))

# Precompute or load embeddings for all table cells
TABLE_CELL_EMB_PATH = os.path.join('data', 'embeddings', 'table_cell_embeddings.pt')
def safe_load_table_cell_embeddings():
    try:
        if os.path.exists(TABLE_CELL_EMB_PATH):
            emb = torch.load(TABLE_CELL_EMB_PATH, map_location='cpu')
            if emb.dtype != torch.float32:
                emb = emb.float()
            print(f'Loaded table cell embeddings from disk. (dtype: {emb.dtype})')
            return emb
        else:
            emb = model.encode(table_cell_texts, convert_to_tensor=True, dtype=torch.float32)
            torch.save(emb, TABLE_CELL_EMB_PATH)
            print(f'Encoded and saved table cell embeddings. (dtype: {emb.dtype})')
            return emb
    except Exception as e:
        print(f'Error loading table cell embeddings: {e}. Regenerating...')
        emb = model.encode(table_cell_texts, convert_to_tensor=True, dtype=torch.float32)
        torch.save(emb, TABLE_CELL_EMB_PATH)
        print(f'Regenerated and saved table cell embeddings. (dtype: {emb.dtype})')
        return emb

table_cell_embeddings = safe_load_table_cell_embeddings()

# --- Clinical Synonym Map ---
age_synonyms = {
    'neonate': ['birth - < 3 months', '<1 month', '0-1 month', 'neonate'],
    'infant': ['1 month to 1 year', '1 month - < 1 year', 'infant', '1 mth to 1 yr', '6 months - <1 year'],
    'toddler': ['1 year - < 6 years', 'toddler', '1-2 yr'],
    'child': ['6 years - < 10 years', 'child', '10 years - < 15 years', '1 year - < 6 years'],
    'adolescent': ['15 years and above', 'adolescent', '10 years - < 15 years'],
    '6 year old': ['6 years - < 10 years', '6-10 years', '6 yr', '6 years'],
    '1 year old': ['1 year - < 6 years', '1 year', '1 yr'],
    'newborn': ['birth - < 3 months', 'neonate', '0-1 month'],
}
param_synonyms = {
    'bp': ['bp', 'blood pressure', 'systolic', 'expected systolic bp (mmhg)', 'minimum systolic bp (mmhg)'],
    'heart rate': ['heart rate', 'pulse'],
    'respiratory rate': ['respiratory rate', 'rr'],
    'urine': ['urine output', 'urine'],
}

def find_synonym_matches(q, lookup_list, synonyms):
    ql = q.lower()
    for key, vals in synonyms.items():
        if key in ql:
            for v in vals:
                for idx, item in enumerate(lookup_list):
                    if v in item.lower():
                        return idx
    return None

@app.post('/search')
async def search(request: QueryRequest):
    try:
        q = request.query.strip()
        threshold = 0.3
        # Try to match age group synonym
        row_labels = [cell[1] for cell in table_cell_lookup]
        col_names = [cell[2] for cell in table_cell_lookup]
        age_idx = find_synonym_matches(q, row_labels, age_synonyms)
        param_idx = find_synonym_matches(q, col_names, param_synonyms)
        filtered_indices = []
        if age_idx is not None or param_idx is not None:
            for idx, cell in enumerate(table_cell_lookup):
                age_match = (age_idx is None or cell[1].lower() == row_labels[age_idx].lower())
                param_match = (param_idx is None or cell[2].lower() == col_names[param_idx].lower())
                if age_match and param_match:
                    filtered_indices.append(idx)
        # If filtered, do semantic search only on those
        if filtered_indices:
            q_emb = model.encode(q, convert_to_tensor=True, dtype=torch.float32)
            filtered_embs = table_cell_embeddings[filtered_indices]
            cos_scores = util.pytorch_cos_sim(q_emb, filtered_embs)[0]
            best_idx = int(torch.argmax(cos_scores))
            best_score = float(cos_scores[best_idx])
            if best_score > threshold:
                real_idx = filtered_indices[best_idx]
                table_title, row_label, col, value = table_cell_lookup[real_idx]
                # --- Natural language answer formatting ---
                # 1. If query is for vital signs (all for age group)
                if any(term in q.lower() for term in ["vital sign", "vitals", "all vital", "normal vital"]):
                    # Find all vital sign columns for this age group
                    vital_cols = [c for c in col_names if any(x in c.lower() for x in ["heart rate", "respiratory rate", "bp", "blood pressure"])]
                    vital_answers = []
                    for idx, cell in enumerate(table_cell_lookup):
                        if cell[1].lower() == row_label.lower() and cell[2] in vital_cols:
                            label = cell[2].replace("(mmHg)", "").replace("Min", "").replace("Max", "").replace(":", "").strip()
                            vital_answers.append(f"{label}: {cell[3]}")
                    if vital_answers:
                        return {"answer": ", ".join(vital_answers)}
                # 2. Otherwise, single parameter answer
                param_label = col.replace("(mmHg)", "").replace(":", "").strip()
                age_label = row_label
                return {"answer": f"The {param_label.lower()} for {age_label} is {value}."}
        # Otherwise, fallback to full semantic search
        q_emb = model.encode(q, convert_to_tensor=True, dtype=torch.float32)
        cos_scores = util.pytorch_cos_sim(q_emb, table_cell_embeddings)[0]
        best_idx = int(torch.argmax(cos_scores))
        best_score = float(cos_scores[best_idx])
        if best_score > threshold:
            table_title, row_label, col, value = table_cell_lookup[best_idx]
            param_label = col.replace("(mmHg)", "").replace(":", "").strip()
            age_label = row_label
            return {"answer": f"The {param_label.lower()} for {age_label} is {value}."}
        # 2. Fallback: semantic search over QA pairs
        if qa_embeddings is not None:
            qa_scores = util.pytorch_cos_sim(q_emb, qa_embeddings)[0]
            qa_best_idx = int(torch.argmax(qa_scores))
            qa_best_score = float(qa_scores[qa_best_idx])
            if qa_best_score > threshold:
                return {"answer": qa_answers[qa_best_idx]}
        # 3. Fallback: return all table titles
        all_titles = '\n'.join([t['title'] for t in all_tables])
        return {"answer": f"Sorry, I couldn't find a specific answer. Available tables are:\n{all_titles}"}
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
