"""
Complete FastAPI Backend for Chatbot
Includes ALL endpoints needed by the frontend + streaming support

Author: Mickey Frankel
Date: 2025-10-30
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncGenerator
from queue import Queue
from threading import Thread
from pathlib import Path
import tempfile
import shutil

# Import your existing system
import sys
sys.path.insert(0, str(Path(__file__).parent))

from database import VectorDatabase
from agent import ChatAgent
from file_loaders import FileLoaderFactory

app = FastAPI(title="Chatbot Complete API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
db = VectorDatabase()
agent = ChatAgent(db)
file_loader = FileLoaderFactory()

# Chat history storage (in-memory)
chat_history = []


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Standard chat endpoint (non-streaming).
    
    This is what the frontend expects by default.
    """
    try:
        # Check if database has documents
        if db.count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents in database. Please upload a CSV file first."
            )
        
        # Get answer from agent
        answer = agent.answer(request.message)
        
        # Store in history
        chat_history.append({
            "role": "user",
            "content": request.message
        })
        chat_history.append({
            "role": "assistant",
            "content": answer
        })
        
        return ChatResponse(response=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint with REAL-TIME thinking display.
    
    Optional alternative to /chat for advanced UIs.
    """
    
    async def generate() -> AsyncGenerator[str, None]:
        """Generate SSE stream of thinking + answer."""
        
        event_queue = Queue()
        answer_container = {'value': None, 'error': None}
        
        def stream_callback(event: dict):
            """Callback that puts events in queue"""
            event_queue.put(event)
        
        def run_agent():
            """Run agent in separate thread"""
            try:
                answer = agent.answer(request.message, stream_func=stream_callback)
                answer_container['value'] = answer
            except Exception as e:
                answer_container['error'] = str(e)
            finally:
                event_queue.put({'type': 'done'})
        
        # Start agent in background thread
        agent_thread = Thread(target=run_agent)
        agent_thread.start()
        
        # Stream events as they arrive
        while True:
            if not event_queue.empty():
                event = event_queue.get()
                
                if event['type'] == 'done':
                    if answer_container['error']:
                        yield f"data: {json.dumps({'type': 'error', 'data': {'message': answer_container['error']}})}\n\n"
                    elif answer_container['value']:
                        yield f"data: {json.dumps({'type': 'answer', 'data': {'text': answer_container['value']}})}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'thinking', 'data': event})}\n\n"
            
            await asyncio.sleep(0.05)
        
        agent_thread.join()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/clear")
async def clear_history():
    """Clear chat history (but keep database)."""
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared"}


@app.post("/clear-database")
async def clear_database():
    """Delete ALL contacts from the database."""
    try:
        count_before = db.count()
        db.reset()
        
        return {
            "message": f"Database cleared. Removed {count_before:,} contacts.",
            "documents_removed": count_before,
            "total_contacts": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process a CSV file."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        
        try:
            # Load documents from CSV
            documents = file_loader.load_file(tmp_path)
            
            if not documents:
                raise HTTPException(
                    status_code=400,
                    detail="No valid data found in CSV file"
                )
            
            # Add to database
            report = db.add_documents(documents)
            
            return {
                "message": f"Successfully uploaded {file.filename}",
                "documents_added": report.documents_added,
                "documents_skipped": report.documents_skipped,
                "total_contacts": report.total_in_db,
                "source": report.source_name
            }
        
        finally:
            # Clean up temp file
            tmp_path.unlink()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    stats = db.get_stats()
    return {
        "document_count": stats.total_documents,
        "sources": stats.sources,
        "total_documents": stats.total_documents
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "documents": db.count()
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Complete API Server...")
    print(f"📊 Loaded {db.count():,} contacts")
    print("🌐 Frontend should connect to: http://localhost:8000")
    print("\n📡 Available endpoints:")
    print("  • POST /chat - Standard chat")
    print("  • POST /chat/stream - Streaming chat with thinking")
    print("  • POST /clear - Clear chat history")
    print("  • POST /clear-database - Delete all data")
    print("  • POST /upload-csv - Upload CSV file")
    print("  • GET /stats - Database statistics")
    print("  • GET /health - Health check\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
