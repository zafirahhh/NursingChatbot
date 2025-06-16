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
import pandas as pd
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
        # Try to find Q: ... A: ... pairs anywhere in the chunk (multi-line, not just at start)
        qa_matches = re.findall(r"Q[:：]\s*(.*?)\nA[:：]\s*([\s\S]*?)(?=\nQ[:：]|$)", chunk, re.IGNORECASE)
        if qa_matches:
            for q, a in qa_matches:
                q = q.strip()
                a = a.strip()
                if q and a:
                    qa_pairs.append((q, a))
                    questions.append(q)
                    answers.append(a)
            continue
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
        # Fallback: If chunk has at least 2 lines, treat first as question, rest as answer
        if len(lines) >= 2:
            q = lines[0]
            a = "\n".join(lines[1:]).strip()
            if q and a:
                qa_pairs.append((q, a))
                questions.append(q)
                answers.append(a)
    return qa_pairs, questions, answers

# Path to your knowledge base text file
KNOWLEDGE_PATH = os.path.join('data', 'nursing_guide_cleaned.txt')

# Load the knowledge base as chunks (paragraphs separated by blank lines)
def load_chunks_from_text(text):
    # Split on double newlines (paragraphs)
    raw_chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    chunks = []
    for chunk in raw_chunks:
        # Split out table sections as separate chunks
        if chunk.startswith('#') and '|' in chunk:
            # Table header and rows
            table_lines = chunk.split('\n')
            for line in table_lines:
                if '|' in line and not line.strip().startswith('#'):
                    # Each table row as a chunk
                    chunks.append(line.strip())
            continue
        # If chunk contains multiple bullet points, split each bullet as a separate chunk
        if re.search(r'\n\s*[•\-\*]', chunk):
            bullets = re.split(r'\n\s*(?=[•\-\*])', chunk)
            for b in bullets:
                b = b.strip()
                if b and len(b.split()) > 4:
                    chunks.append(b)
            continue
        # If chunk contains numbered list, split each number as a chunk
        if re.search(r'\n\s*\d+\.', chunk):
            numbers = re.split(r'\n\s*(?=\d+\.)', chunk)
            for n in numbers:
                n = n.strip()
                if n and len(n.split()) > 4:
                    chunks.append(n)
            continue
        # Otherwise, treat the paragraph as a chunk if it's not too short
        if len(chunk.split()) > 4:
            chunks.append(chunk)
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
            emb = torch.load(path, map_location='cpu', weights_only=True)
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

# --- Embedding Model ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load knowledge base
# (Refined chunking logic already present)

def embed_chunks(chunks):
    return model.encode(chunks, convert_to_tensor=True)

def find_best_answer(user_query, chunks, chunk_embeddings, top_k=5):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)[0]
    re_ranked = sorted(hits, key=lambda x: float(x['score']), reverse=True)
    for hit in re_ranked:
        if float(hit['score']) > 0.4:  # threshold to prevent bad matches
            return chunks[hit['corpus_id']]
    return "Sorry, I could not find relevant information for that question."

# Read knowledge base
with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
    text = f.read()

chunks = load_chunks_from_text(text)
chunk_embeddings = embed_chunks(chunks)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    answer = find_best_answer(query.question, chunks, chunk_embeddings)
    return {"answer": answer}

class QueryRequest(BaseModel):
    query: str

# --- Structured Table Extraction ---
# DISABLE extract_all_pipe_tables and all_tables since we now use nursing_guide_cleaned.txt and vitals_df
# def extract_all_pipe_tables():
#     table_path = os.path.join('data', 'nursing_guide.txt')
#     with open(table_path, encoding='utf-8') as f:
#         lines = f.readlines()
#     tables = []
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         if line.strip().startswith('#') and '|' in line:
#             header = [h.strip('# ').strip() for h in line.strip().split('|')]
#             table_rows = []
#             i += 1
#             while i < len(lines) and '|' in lines[i]:
#                 row = [col.strip() for col in lines[i].strip().split('|')]
#                 if len(row) == len(header):
#                     table_rows.append(dict(zip(header, row)))
#                 i += 1
#             tables.append({'header': header, 'rows': table_rows, 'title': line.strip('#').strip()})
#         else:
#             i += 1
#     return tables

def extract_all_pipe_tables():
    table_path = os.path.join('data', 'nursing_guide_cleaned.txt')
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
            emb = torch.load(TABLE_ROW_EMB_PATH, map_location='cpu', weights_only=True)
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
            emb = torch.load(TABLE_CELL_EMB_PATH, map_location='cpu', weights_only=True)
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

# --- Row-level Embedding Preparation ---
row_texts = []  # List of row text for semantic search
row_lookup = []
for table in all_tables:
    for row in table['rows']:
        # Join all values for semantic search, include table title and column names for context
        row_text = f"Table: {table['title']} | " + ' | '.join([f"{col}: {row[col]}" for col in table['header']])
        row_texts.append(row_text)
        row_lookup.append((table['title'], row))

ROW_EMB_PATH = os.path.join('data', 'embeddings', 'row_embeddings.pt')
def safe_load_row_embeddings():
    try:
        if os.path.exists(ROW_EMB_PATH):
            emb = torch.load(ROW_EMB_PATH, map_location='cpu', weights_only=True)
            if emb.dtype != torch.float32:
                emb = emb.float()
            print(f'Loaded row embeddings from disk. (dtype: {emb.dtype})')
            return emb
        else:
            emb = model.encode(row_texts, convert_to_tensor=True, dtype=torch.float32)
            torch.save(emb, ROW_EMB_PATH)
            print(f'Encoded and saved row embeddings. (dtype: {emb.dtype})')
            return emb
    except Exception as e:
        print(f'Error loading row embeddings: {e}. Regenerating...')
        emb = model.encode(row_texts, convert_to_tensor=True, dtype=torch.float32)
        torch.save(emb, ROW_EMB_PATH)
        print(f'Regenerated and saved row embeddings. (dtype: {emb.dtype})')
        return emb

row_embeddings = safe_load_row_embeddings()

# --- Clinical Synonym Map ---
age_synonyms = {
    'neonate': ['birth - < 3 months', '<1 month', '0-1 month', 'neonate', 'newborn', '0 to 1 month', '0–1 month'],
    'infant': ['1 month to 1 year', '1 month - < 1 year', 'infant', '1 mth to 1 yr', '6 months - <1 year', '1-12 months', '1 to 12 months'],
    'toddler': ['1 year - < 6 years', 'toddler', '1-2 yr', '1-2 years', '1 to 2 years'],
    'child': ['6 years - < 10 years', 'child', '10 years - < 15 years', '1 year - < 6 years', '1-10 years', '1 to 10 years', '2-12 years', '2 to 12 years'],
    'adolescent': ['15 years and above', 'adolescent', '10 years - < 15 years', '12-18 years', '12 to 18 years'],
    '6 year old': ['6 years - < 10 years', '6-10 years', '6 yr', '6 years'],
    '1 year old': ['1 year - < 6 years', '1 year', '1 yr'],
    'newborn': ['birth - < 3 months', 'neonate', '0-1 month', '0 to 1 month'],
    'young child': ['young child', '2-7 years', '2 - 7 years', 'younger child', 'child (2-7 years)', 'young children'],
    'older child': ['older child', '7-12 years', '7 - 12 years', 'child (7-12 years)', 'older children'],
}
param_synonyms = {
    'bp': ['bp', 'blood pressure', 'systolic', 'expected systolic bp (mmhg)', 'minimum systolic bp (mmhg)', 'systolic blood pressure', 'hypotension'],
    'heart rate': ['heart rate', 'pulse'],
    'respiratory rate': ['respiratory rate', 'rr'],
    'urine': ['urine output', 'urine'],
    'formula': ['formula', 'calculation', 'calculate', 'expected', 'range'],
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

# --- Vitals Table Parsing ---
def load_vitals_table():
    vitals_path = os.path.join('data', 'nursing_guide_cleaned.txt')
    # Read only the first table (assume it's at the top, separated by blank lines)
    with open(vitals_path, encoding='utf-8') as f:
        lines = f.readlines()
    table_lines = []
    for line in lines:
        if '|' in line:
            table_lines.append(line)
        elif table_lines:
            break  # Stop at first blank line after table
    # Write to a temp string and read as CSV
    import io
    table_str = ''.join(table_lines)
    df = pd.read_csv(io.StringIO(table_str), sep='|')
    # Clean columns and values
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df
vitals_df = load_vitals_table()

# --- Age alias map for vitals table ---
vitals_age_aliases = {
    'neonate': ['0-1 month', 'neonate', 'newborn'],
    'infant': ['1 month - 1 year', '1 month to 1 year', 'infant'],
    'toddler': ['1-2 years', 'toddler'],
    'child': ['2-12 years', 'child'],
    'adolescent': ['12-18 years', 'adolescent'],
    'young child': ['2-7 years', 'young child', 'younger child'],
    'older child': ['7-12 years', 'older child'],
}

def find_vitals_row_for_age(query):
    ql = query.lower()
    # 1. Try exact match first
    for idx, row in vitals_df.iterrows():
        age_val = row[vitals_df.columns[0]].lower().strip()
        if age_val == ql:
            return age_val, row
    # 2. Try longest substring match (most specific)
    best_match = None
    best_len = 0
    for idx, row in vitals_df.iterrows():
        age_val = row[vitals_df.columns[0]].lower().strip()
        if age_val in ql or ql in age_val:
            if len(age_val) > best_len:
                best_match = (age_val, row)
                best_len = len(age_val)
    if best_match:
        return best_match
    # 3. Try alias/synonym match
    for alias, options in vitals_age_aliases.items():
        for opt in options:
            if opt in ql or alias in ql:
                for idx, row in vitals_df.iterrows():
                    age_val = row[vitals_df.columns[0]].lower()
                    if opt in age_val or alias in age_val:
                        return alias, row
    # 4. Fallback: try partial match
    for idx, row in vitals_df.iterrows():
        age_val = row[vitals_df.columns[0]].lower()
        if any(a in ql for a in age_val.split()):
            return age_val, row
    return None, None

def normalize(s):
    s = s.lower()
    s = re.sub(r'\([^)]*\)', '', s)
    s = s.replace('-', ' ').replace('<', '').replace('>', '').replace(':', '').replace('/', ' ')
    s = re.sub(r'[^a-z0-9 ]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

@app.post('/search')
async def search(request: QueryRequest):
    try:
        q = request.query.strip()
        ql = q.lower()
        threshold = 0.38  # Lowered threshold for semantic search
        q_emb = model.encode(q, convert_to_tensor=True, dtype=torch.float32)
        best_age_label = None
        best_param_label = None
        # --- 1. Vitals/Table Logic FIRST ---
        vitals_keywords = ["vital sign", "vitals", "normal vital"]
        specific_vital_params = ["heart rate", "respiratory rate", "systolic bp", "blood pressure", "bp"]
        min_synonyms = ["minimum", "min", "lowest", "lower"]
        max_synonyms = ["maximum", "max", "highest", "upper"]
        age_label, row = find_vitals_row_for_age(ql)
        param_found = any(param in ql for param in specific_vital_params)
        if row is not None and param_found:
            for param in specific_vital_params:
                if param in ql:
                    is_min = any(word in ql for word in min_synonyms)
                    is_max = any(word in ql for word in max_synonyms)
                    matching_cols = [c for c in vitals_df.columns[1:] if param.replace("bp", "blood pressure") in c or c in param]
                    if matching_cols:
                        answers = []
                        for c in matching_cols:
                            c_norm = c.lower()
                            val = row[c]
                            if val and val.lower() != 'nan':
                                if is_min and "min" in c_norm:
                                    answers.append(f"For {age_label}, the minimum {param} is {val}.")
                                elif is_max and "max" in c_norm:
                                    answers.append(f"For {age_label}, the maximum {param} is {val}.")
                                elif not is_min and not is_max:
                                    answers.append(f"For {age_label}, the {param} is {val}.")
                        if answers:
                            return {"answer": ' '.join(answers)}
        elif row is not None:
            # Only age matched, return all vitals for that age
            col_map = {c: c for c in vitals_df.columns if c != vitals_df.columns[0]}
            col_map = {c: c.replace('bpm', 'beats per minute').replace('respiratory rate', 'respiratory rate').replace('heart rate', 'heart rate').replace('bp', 'blood pressure') for c in col_map}
            values = []
            for c in vitals_df.columns[1:]:
                val = row[c]
                if val and val.lower() != 'nan':
                    values.append(f"{col_map[c]} is {val}")
            age_phrase = age_label if age_label else row[vitals_df.columns[0]]
            return {"answer": f"For {age_phrase} ({row[vitals_df.columns[0]]}), " + ' and '.join(values) + "."}
        elif param_found:
            # Only parameter matched, return values for all age groups
            param = next((p for p in specific_vital_params if p in ql), None)
            if param:
                matching_cols = [c for c in vitals_df.columns[1:] if param.replace("bp", "blood pressure") in c or c in param]
                if matching_cols:
                    answers = []
                    for idx, r in vitals_df.iterrows():
                        age = r[vitals_df.columns[0]]
                        for c in matching_cols:
                            val = r[c]
                            if val and val.lower() != 'nan':
                                answers.append(f"For {age}, the {param} is {val}.")
                    if answers:
                        return {"answer": ' '.join(answers)}
        # --- 2. Direct Table Cell Answer for Specific Parameter ---
        matched_age = None
        matched_param = None
        best_age_score = 0
        best_param_score = 0
        matched_age_syn = None
        matched_param_syn = None
        for key, vals in age_synonyms.items():
            for v in vals:
                v_norm = normalize(v)
                if v_norm in normalize(ql) or normalize(ql) in v_norm:
                    for cell in table_cell_lookup:
                        row_norm = normalize(cell[1])
                        if v_norm == row_norm:
                            matched_age = cell[1]
                            matched_age_syn = v
                            best_age_score = 999
                        elif v_norm in row_norm or row_norm in v_norm:
                            score = len(v_norm)
                            if score > best_age_score:
                                matched_age = cell[1]
                                matched_age_syn = v
                                best_age_score = score
        for key, vals in param_synonyms.items():
            for v in vals:
                v_norm = normalize(v)
                if v_norm in normalize(ql) or normalize(ql) in v_norm:
                    for cell in table_cell_lookup:
                        col_norm = normalize(cell[2])
                        if v_norm == col_norm:
                            matched_param = cell[2]
                            matched_param_syn = v
                            best_param_score = 999
                        elif v_norm in col_norm or col_norm in v_norm:
                            score = len(v_norm)
                            if score > best_param_score:
                                matched_param = cell[2]
                                matched_param_syn = v
                                best_param_score = score
        if matched_age:
            best_age_label = matched_age
        if matched_param:
            best_param_label = matched_param
        if matched_age and matched_param:
            best_cell = None
            for cell in table_cell_lookup:
                if normalize(cell[1]) == normalize(matched_age) and normalize(cell[2]) == normalize(matched_param):
                    best_cell = cell
                    break
            if best_cell:
                param_label = best_cell[2].replace("(mmHg)", "").replace(":", "").strip()
                age_label = best_cell[1]
                value = best_cell[3]
                return {"answer": f"For {age_label}, the normal {param_label.lower()} is {value}."}
        # --- 3. QA Pair Semantic Search (for non-vitals/table queries) ---
        improved_threshold = 0.55  # Only accept strong matches
        if qa_embeddings is not None:
            qa_scores = util.pytorch_cos_sim(q_emb, qa_embeddings)[0]
            qa_best_idx = int(torch.argmax(qa_scores))
            qa_best_score = float(qa_scores[qa_best_idx])
            qa_answer = qa_answers[qa_best_idx]
            # Filter out copyright/disclaimer, too short, or generic answers
            # Lowered threshold for QA semantic search
            if qa_best_score > 0.45 and len(qa_answer.split()) > 6 and not re.search(r'copyright|distribution is allowed|worldscientific|KK Women|The Baby Bear Book', qa_answer, re.I):
                return {"answer": qa_answer}
        # --- 4. Chunk Semantic Search (extract best sentence) ---
        chunk_cos_scores = util.pytorch_cos_sim(q_emb, chunk_embeddings)[0]
        best_chunk_idx = int(torch.argmax(chunk_cos_scores))
        best_chunk_score = float(chunk_cos_scores[best_chunk_idx])
        # Lowered threshold for chunk semantic search
        if best_chunk_score > 0.40:
            chunk = chunks[best_chunk_idx]
            sentences = sent_tokenize(chunk)
            # Filter out copyright/disclaimer, too short, or generic sentences
            sentences = [s for s in sentences if len(s.split()) > 6 and not re.search(r'copyright|distribution is allowed|worldscientific|KK Women|The Baby Bear Book', s, re.I)]
            if not sentences:
                return {"answer": "Sorry, no relevant answer found."}
            sent_embs = model.encode(sentences, convert_to_tensor=True, dtype=torch.float32)
            sent_scores = util.pytorch_cos_sim(q_emb, sent_embs)[0]
            best_sent_idx = int(torch.argmax(sent_scores))
            best_sent_score = float(sent_scores[best_sent_idx])
            best_sent = sentences[best_sent_idx]
            # Lowered threshold for best sentence
            if best_sent_score > 0.38:
                return {"answer": best_sent}
        # --- 5. Fallback: Table Row/Cell/QA Semantic Search (as before) ---
        # New: Fallback to most relevant chunk or table row that matches at least one keyword from the question
        import heapq
        import string
        def extract_keywords(text):
            # Remove punctuation and split
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = set(w.lower() for w in text.split() if len(w) > 2)
            return words
        q_keywords = extract_keywords(q)
        # Search chunks for keyword overlap
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_words = extract_keywords(chunk)
            if q_keywords & chunk_words:
                score = float(util.pytorch_cos_sim(q_emb, model.encode([chunk], convert_to_tensor=True, dtype=torch.float32))[0][0])
                chunk_scores.append((score, chunk))
        if chunk_scores:
            best_score, best_chunk = max(chunk_scores)
            return {"answer": best_chunk}
        # Search table rows for keyword overlap
        row_scores = []
        for i, row in enumerate(table_row_texts):
            row_words = extract_keywords(row)
            if q_keywords & row_words:
                score = float(util.pytorch_cos_sim(q_emb, model.encode([row], convert_to_tensor=True, dtype=torch.float32))[0][0])
                row_scores.append((score, row))
        if row_scores:
            best_score, best_row = max(row_scores)
            return {"answer": best_row}
        fallback_age = best_age_label if best_age_label else "the specified age group"
        fallback_param = best_param_label if best_param_label else "the specific parameter"
        return {"answer": f"Sorry, the specific range for {fallback_age} and {fallback_param} is not available."}
    except Exception as e:
        import traceback
        print(f"Error in /search: {e}")
        traceback.print_exc()
        return {"answer": "Internal server error. Please check backend logs."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
