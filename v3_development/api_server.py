"""
FastAPI Backend with REAL Streaming Thinking Display

This provides an API endpoint that streams agent thinking in REAL-TIME
to the frontend as events happen.

Author: Mickey Frankel
Date: 2025-10-30
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncGenerator
from queue import Queue
from threading import Thread

# Import your existing system
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database import VectorDatabase
from agent import ChatAgent

app = FastAPI(title="Chatbot with Real-Time Thinking")

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


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint with REAL-TIME thinking display.
    
    Streams events as they happen:
      - type: "thinking" - Agent reasoning
      - type: "answer" - Final answer
      - type: "done" - Stream complete
    """
    
    async def generate() -> AsyncGenerator[str, None]:
        """Generate SSE stream of thinking + answer."""
        
        # Queue for thread-safe communication
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
            # Check queue for new events
            if not event_queue.empty():
                event = event_queue.get()
                
                if event['type'] == 'done':
                    # Send final answer
                    if answer_container['error']:
                        yield f"data: {json.dumps({'type': 'error', 'data': {'message': answer_container['error']}})}\n\n"
                    elif answer_container['value']:
                        yield f"data: {json.dumps({'type': 'answer', 'data': {'text': answer_container['value']}})}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    break
                else:
                    # Stream thinking event
                    yield f"data: {json.dumps({'type': 'thinking', 'data': event})}\n\n"
            
            await asyncio.sleep(0.05)  # Small delay to prevent busy waiting
        
        agent_thread.join()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    stats = db.get_stats()
    return stats.to_dict()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "documents": db.count()}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting streaming API server...")
    print("📡 Thinking events will stream in real-time!")
    print("🌐 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
