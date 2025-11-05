# Dog Breed Search - Project Structure

## 📁 Folder Structure

```
dog-breed-search/
├── frontend/              # React frontend (create with create-react-app or Vite)
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   │   └── api.js     # WebSocket & REST API client
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/               # All backend code (Python + API + Config + Data)
│   ├── app.py             # FastAPI server with WebSocket
│   ├── complete_search_engine.py
│   ├── ingestion_pipeline.py
│   ├── llm_query_parser.py
│   ├── search_cli.py
│   ├── main.py            # Ingestion pipeline entry point
│   ├── requirements.txt   # Python dependencies
│   ├── routes/            # API route modules (for future expansion)
│   ├── config/            # Configuration files
│   │   └── .env           # API keys and secrets
│   ├── data/              # Data files
│   │   └── enriched_breeds_*.json
│   └── logs/              # Log files
│
└── venv/                  # Python virtual environment
```

## 🔄 Architecture Flow

```
┌─────────────────┐
│  React Frontend │
│  (frontend/)    │
└────────┬────────┘
         │
         │ WebSocket / REST API
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (backend/app.py)│
└────────┬────────┘
         │
         │ Uses
         ▼
┌─────────────────┐
│  Search Engine  │
│  (backend/*.py) │
└────────┬────────┘
         │
         │ Connects to
         ▼
┌─────────────────┐
│  Pinecone       │
│  OpenAI         │
└─────────────────┘
```

## 🚀 Quick Start

### 1. Backend API
```bash
cd backend
pip install -r requirements.txt
python app.py
# Server runs on http://localhost:8000
```

### 2. Ingestion Pipeline
```bash
cd backend
python main.py
```

### 3. Frontend (to be created)
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:3000
```

## 📡 API Endpoints

### Health Check (HTTP)
- `GET /` - Server status
- `GET /health` - Health check

### WebSocket (Search)
- `ws://localhost:8000/ws/search` - Real-time search with progress updates

## 🔌 WebSocket Message Format

### Client → Server
```json
{
    "type": "search",
    "query": "small friendly dog",
    "top_k": 10,
    "rerank_top_n": 50
}
```

### Server → Client
```json
// Progress
{
    "type": "progress",
    "stage": "query_parsing",
    "message": "Understanding your query..."
}

// Results
{
    "type": "results",
    "data": {
        "results": [...],
        "metadata": {...}
    }
}

// Error
{
    "type": "error",
    "message": "Error description"
}
```

## 📝 File Paths

All paths are relative to the `backend/` directory:
- Config: `config/.env`
- Data: `data/enriched_breeds_*.json`
- Logs: `logs/`
