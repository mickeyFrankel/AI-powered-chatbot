#!/usr/bin/env python3
"""
FastAPI backend for chatbot
"""
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vectoric_search import AdvancedVectorDBQASystem
import uvicorn

app = FastAPI(title="Chatbot API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QA system
qa_system = None

@app.on_event("startup")
async def startup():
    global qa_system
    qa_system = AdvancedVectorDBQASystem(persist_directory="./chroma_db")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    stats: dict = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat message with timeout"""
    if not qa_system:
        raise HTTPException(status_code=500, detail="QA system not initialized")
    
    try:
        # Run with 30 second timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(qa_system.agent_answer, request.message),
            timeout=30.0
        )
        return ChatResponse(response=response)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out. Please try a simpler query.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if not qa_system:
        raise HTTPException(status_code=500, detail="QA system not initialized")
    
    return qa_system.get_collection_stats()

@app.post("/clear")
async def clear_history():
    """Clear conversation history"""
    if not qa_system:
        raise HTTPException(status_code=500, detail="QA system not initialized")
    
    qa_system._clear_history()
    return {"message": "History cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
