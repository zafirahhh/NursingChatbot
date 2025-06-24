# KKH Nursing Chatbot

A FastAPI-based chatbot for nursing guidance using semantic search and quiz functionality.

## Quick Start Options

### Option 1: One-Click Local Setup (Windows)
1. Download the project
2. Double-click `start_chatbot.bat`
3. Open `index.html` in your browser

### Option 2: Deploy to Railway (Recommended for sharing)
1. Push code to GitHub
2. Connect to [Railway](https://railway.app)
3. Deploy automatically - no setup needed!

### Option 3: Docker (Cross-platform)
```bash
docker build -t kkh-chatbot .
docker run -p 8000:8000 kkh-chatbot
```

### Option 4: Manual Setup
```bash
# Create virtual environment
python -m venv eb-env
source eb-env/bin/activate  # On Windows: eb-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

## Deployment URLs
- **Local**: http://localhost:8000
- **Railway**: Your deployed URL will be provided after deployment

## Features
- 🤖 AI-powered nursing guidance
- 📚 Knowledge base search
- 🧠 Interactive quizzes
- 💬 Session management

## Tech Stack
- **Backend**: FastAPI, SentenceTransformers
- **Frontend**: Vanilla JavaScript, HTML/CSS
- **ML**: PyTorch, NLTK
