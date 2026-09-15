#!/usr/bin/env python3
"""
FastAPI backend for chatbot
"""
import os
import asyncio
import shutil
import time
from datetime import date
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from vectoric_search import AdvancedVectorDBQASystem
import uvicorn

app = FastAPI(title="Chatbot API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple global daily rate-limit backstop for the public demo ---
# Not per-user auth - just a cost/abuse ceiling since this is link-only,
# unauthenticated, and hitting a real LLM API. Resets on process restart
# too, which only makes it more conservative, never less.
DAILY_LIMIT = int(os.environ.get("DAILY_CHAT_LIMIT", "200"))
_rate_state = {"day": date.today().isoformat(), "count": 0}

def _check_and_consume_quota():
    today = date.today().isoformat()
    if _rate_state["day"] != today:
        _rate_state["day"] = today
        _rate_state["count"] = 0
    if _rate_state["count"] >= DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Demo has hit its daily query limit - please try again tomorrow, or reach out and I'll bump the cap."
        )
    _rate_state["count"] += 1

# Initialize QA system
qa_system = None

@app.on_event("startup")
async def startup():
    global qa_system
    # Updated path for reorganized structure
    db_path = os.path.join(os.path.dirname(__file__), "../../databases/vector/chroma_db")
    qa_system = AdvancedVectorDBQASystem(persist_directory=db_path)

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

    _check_and_consume_quota()

    try:
        # Run with 60 second timeout (increased for complex queries)
        response = await asyncio.wait_for(
            asyncio.to_thread(qa_system.agent_answer, request.message),
            timeout=60.0
        )
        return ChatResponse(response=response)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Query took too long. Try breaking it into simpler questions."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

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

@app.post("/clear-database")
async def clear_database():
    """Clear entire database (all contacts)"""
    global qa_system
    
    if not qa_system:
        raise HTTPException(status_code=500, detail="QA system not initialized")
    
    # Check if database is already empty
    current_count = qa_system.collection.count()
    if current_count == 0:
        return {"message": "Database is already empty.", "success": True}
    
    try:
        # Delete physical directories
        db_path = os.path.join(os.path.dirname(__file__), "../../databases/vector/chroma_db")
        contacts_db_path = os.path.join(os.path.dirname(__file__), "../../databases/contacts_db")
        
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        if os.path.exists(contacts_db_path):
            shutil.rmtree(contacts_db_path)
        
        # Create fresh QA system
        qa_system = AdvancedVectorDBQASystem(persist_directory=db_path)
        
        return {"message": "Database cleared successfully. Ready for new data.", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process a CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    temp_path = f"./temp_upload_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the CSV (preprocessing + phone number fixing)
        if qa_system:
            ingestion_report = qa_system.ingest_file(temp_path)
        else:
            raise HTTPException(status_code=500, detail="QA system not initialized")
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "message": f"Successfully loaded {file.filename}",
            "documents_added": ingestion_report['documents_added'],
            "total_contacts": ingestion_report['total_in_db']
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# Serve the built React frontend (same origin as the API - simplest
# possible setup for a single-container HF Space deployment).
_frontend_dist = os.path.join(os.path.dirname(__file__), "../../frontend/dist")
if os.path.isdir(_frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Let API routes above take precedence; anything else falls
        # through to the SPA's index.html (client-side routing safe).
        return FileResponse(os.path.join(_frontend_dist, "index.html"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
