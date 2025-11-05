"""
FastAPI Server for Dog Breed Search
WebSocket-only API for real-time search with progress updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# Import from same directory (all backend code is here now)
from complete_search_engine import CompleteSearchEngine

app = FastAPI(
    title="Dog Breed Search API",
    description="WebSocket API for intelligent dog breed search with real-time progress updates",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine (singleton)
search_engine = None

# Thread pool executor for running synchronous search operations
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    global search_engine
    print("🚀 Initializing Dog Breed Search Engine...")
    search_engine = CompleteSearchEngine(
        use_llm_parser=True,
        use_reranking=True,
        use_post_filtering=True
    )
    print("✅ Search engine ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Dog Breed Search API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "engine_ready": search_engine is not None}

@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dog breed search
    
    Supports:
    - Real-time search updates
    - Progress notifications
    - Streaming results
    """
    await websocket.accept()
    
    if not search_engine:
        await websocket.send_json({
            "type": "error",
            "message": "Search engine not initialized"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "search":
                query = data.get("query", "")
                top_k = data.get("top_k", 24)  # Default to 24 for pagination (12 per page)
                rerank_top_n = data.get("rerank_top_n", 100)  # Increased for better quality with 24 results
                
                if not query:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Query is required"
                    })
                    continue
                
                # Send search started
                await websocket.send_json({
                    "type": "search_started",
                    "query": query
                })
                
                # Perform search with progress updates
                try:
                    # Stage 1: Query parsing
                    await websocket.send_json({
                        "type": "progress",
                        "stage": "query_parsing",
                        "message": "Understanding your query..."
                    })
                    
                    # Run LLM parsing in thread pool to avoid blocking
                    llm_filters = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: search_engine.llm_parser.parse(query, verbose=False)
                    )
                    
                    # Stage 2: Vector search
                    await websocket.send_json({
                        "type": "progress",
                        "stage": "vector_search",
                        "message": "Searching database...",
                        "filters": llm_filters
                    })
                    
                    # Perform full search in thread pool to avoid blocking
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: search_engine.search(
                            query=query,
                            top_k=top_k,
                            rerank_top_n=rerank_top_n,
                            verbose=False
                        )
                    )
                    
                    # Send results
                    await websocket.send_json({
                        "type": "results",
                        "data": response
                    })
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"Search error: {error_trace}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Search failed: {str(e)}",
                        "traceback": error_trace
                    })
            
            elif data.get("type") == "ping":
                # Health check
                await websocket.send_json({
                    "type": "pong",
                    "status": "connected"
                })
            
            elif data.get("type") == "close":
                # Client requested disconnect
                break
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"WebSocket error: {error_trace}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "traceback": error_trace
            })
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Configure logging to both file and console
    import os
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/server.log'),
            logging.StreamHandler()
        ]
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)

