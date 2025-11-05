# Dog Breed Search API

FastAPI server with WebSocket-only API for dog breed search with real-time progress updates.

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
# OR
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

### Health Check (HTTP)

- `GET /` - Health check
- `GET /health` - Health status

### WebSocket (Search)

- `ws://localhost:8000/ws/search` - WebSocket search endpoint (primary)

## WebSocket Usage

### Connect
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/search');
```

### Send Search Request
```javascript
ws.send(JSON.stringify({
    type: "search",
    query: "small friendly dog",
    top_k: 10,
    rerank_top_n: 50
}));
```

### Receive Messages

**Progress Update:**
```json
{
    "type": "progress",
    "stage": "query_parsing",
    "message": "Understanding your query..."
}
```

**Results:**
```json
{
    "type": "results",
    "data": {
        "results": [...],
        "metadata": {...}
    }
}
```

**Error:**
```json
{
    "type": "error",
    "message": "Error message"
}
```

## Environment Variables

Make sure `config/.env` has:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `DOGS_API_URL`

