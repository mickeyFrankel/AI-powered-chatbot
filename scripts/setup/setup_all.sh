#!/bin/bash
# COMPLETE CHATBOT SETUP - Run this in your chatbot directory
# chmod +x setup_all.sh && ./setup_all.sh

set -e
echo "🚀 Creating all production files..."

# ===== 1. CREATE app.py =====
cat > app.py << 'APPPY_EOF'
import os, sys, logging, json, asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('chatbot_api.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

try:
    from vectoric_search import AdvancedVectorDBQASystem
    VECTORDB_AVAILABLE = True
    logger.info("✅ VectorDB imported")
except ImportError as e:
    VECTORDB_AVAILABLE = False
    logger.warning(f"⚠️ VectorDB not found: {e}")

class Config:
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    VECTORDB_PERSIST_DIR = os.getenv("VECTORDB_PERSIST_DIR", "./chroma_db")
    VECTORDB_MODEL_NAME = os.getenv("VECTORDB_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_CALLS = int(os.getenv("RATE_LIMIT_CALLS", "10"))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "20"))
    DEFAULT_RESULTS = int(os.getenv("DEFAULT_RESULTS", "10"))

app = FastAPI(title="Chatbot API with Reasoning", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=Config.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

class RateLimiter:
    def __init__(self):
        self.requests = {}
    async def check_rate_limit(self, client_ip: str) -> bool:
        if not Config.RATE_LIMIT_ENABLED:
            return True
        now = datetime.now()
        self.requests = {ip: times for ip, times in self.requests.items() if any((now - t).total_seconds() < Config.RATE_LIMIT_PERIOD for t in times)}
        recent = [t for t in self.requests.get(client_ip, []) if (now - t).total_seconds() < Config.RATE_LIMIT_PERIOD]
        if len(recent) >= Config.RATE_LIMIT_CALLS:
            return False
        recent.append(now)
        self.requests[client_ip] = recent
        return True

rate_limiter = RateLimiter()
qa_system = None

def initialize_qa_system():
    global qa_system
    if not VECTORDB_AVAILABLE:
        return None
    try:
        logger.info("Initializing VectorDB...")
        qa_system = AdvancedVectorDBQASystem(persist_directory=Config.VECTORDB_PERSIST_DIR, model_name=Config.VECTORDB_MODEL_NAME)
        logger.info(f"✅ QA System ready with {qa_system.collection.count()} docs")
        return qa_system
    except Exception as e:
        logger.error(f"❌ Init failed: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting API...")
    initialize_qa_system()

class ReasoningStreamer:
    @staticmethod
    async def stream_search_reasoning(query: str, max_results: int, qa_system_instance) -> AsyncGenerator[str, None]:
        try:
            yield ReasoningStreamer._format_sse({'type': 'reasoning_start', 'message': 'Starting search...', 'timestamp': datetime.now().isoformat()})
            await asyncio.sleep(0.05)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, qa_system_instance.comprehensive_search, query, max_results, True)
            for step in results.get('reasoning', []):
                yield ReasoningStreamer._format_sse({'type': 'reasoning_step', 'step': step})
                await asyncio.sleep(0.08)
            yield ReasoningStreamer._format_sse({'type': 'results', 'data': {'query': results['query'], 'entity': results['entity'], 'total_found': results['total_found'], 'results': [{'id': r.get('id'), 'name': r.get('name'), 'phone': r.get('metadata', {}).get('phone', 'N/A'), 'score': round(r.get('score', 0), 1), 'methods': r.get('methods', [])} for r in results['results']]}})
            yield ReasoningStreamer._format_sse({'type': 'reasoning_complete', 'message': 'Complete', 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            yield ReasoningStreamer._format_sse({'type': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()})
    
    @staticmethod
    def _format_sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html><html><head><title>Chatbot API</title><style>body{font-family:Arial;max-width:800px;margin:50px auto;padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white}.container{background:white;color:#333;padding:40px;border-radius:15px;box-shadow:0 10px 40px rgba(0,0,0,0.3)}h1{color:#667eea}a{color:#667eea;font-weight:bold}</style></head><body><div class="container"><h1>🤖 Chatbot API</h1><p>✅ Running!</p><ul><li><a href="/ui">🌐 Web UI</a></li><li><a href="/api/docs">📚 API Docs</a></li><li><a href="/api/health">🏥 Health</a></li></ul></div></body></html>"""

@app.get("/ui", response_class=FileResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "reasoning_web_ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(ui_path)

@app.get("/api/health")
async def health_check():
    total_docs = None
    if qa_system:
        try:
            total_docs = qa_system.collection.count()
        except: pass
    return {"status": "healthy" if VECTORDB_AVAILABLE else "degraded", "timestamp": datetime.now().isoformat(), "vectordb_available": VECTORDB_AVAILABLE and qa_system is not None, "total_documents": total_docs}

@app.get("/api/search")
async def search_stream(request: Request, query: str, show_reasoning: bool = True, max_results: Optional[int] = None):
    client_ip = request.client.host
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    if not query or not query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query required")
    if max_results is None:
        max_results = Config.DEFAULT_RESULTS
    if not qa_system:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Search unavailable")
    logger.info(f"Search: {client_ip} - '{query}'")
    return StreamingResponse(ReasoningStreamer.stream_search_reasoning(query, max_results, qa_system), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Chatbot API with Reasoning")
    print("="*60)
    print(f"\n📍 API: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"🌐 UI: http://{Config.API_HOST}:{Config.API_PORT}/ui")
    print(f"📚 Docs: http://{Config.API_HOST}:{Config.API_PORT}/api/docs")
    print(f"\n⚙️ VectorDB: {'✅ Available' if VECTORDB_AVAILABLE else '❌ Demo Mode'}")
    print("\n" + "="*60 + "\n")
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT, log_level="info")
APPPY_EOF

echo "✅ Created app.py"

# Create other files (env, requirements, start.sh, HTML)
cat > .env << 'ENV_EOF'
OPENAI_API_KEY=your-openai-key-here
API_HOST=0.0.0.0
API_PORT=8000
VECTORDB_PERSIST_DIR=./chroma_db
VECTORDB_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV_EOF

cat > requirements.txt << 'REQ_EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
chromadb==0.4.18
sentence-transformers==2.2.2
pandas numpy langchain langchain-openai rapidfuzz openpyxl python-docx PyPDF2 scikit-learn
REQ_EOF

cat > start.sh << 'START_EOF'
#!/bin/bash
set -e
source .venv/bin/activate 2>/dev/null || (python3 -m venv .venv && source .venv/bin/activate)
pip install -q -r requirements.txt
python app.py
START_EOF
chmod +x start.sh

cat > reasoning_web_ui.html << 'HTML_EOF'
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>Chatbot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;padding:20px}
.container{max-width:900px;margin:0 auto}
.header{text-align:center;color:white;margin-bottom:30px}
.header h1{font-size:2.5em}
.search-box{background:white;border-radius:15px;padding:30px;margin-bottom:30px}
#queryInput{flex:1;padding:15px;font-size:16px;border:2px solid #e0e0e0;border-radius:10px;width:70%}
#searchBtn{padding:15px 40px;font-size:16px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:none;border-radius:10px;cursor:pointer}
.reasoning-container{background:white;border-radius:15px;padding:30px;margin-bottom:30px;display:none}
.reasoning-container.show{display:block}
.reasoning-step{border:1px solid #e0e0e0;border-radius:10px;margin-bottom:10px;padding:15px;background:#f8f9fa}
.results-container{background:white;border-radius:15px;padding:30px;display:none}
.results-container.show{display:block}
.result-card{border:1px solid #e0e0e0;border-radius:10px;padding:20px;margin-bottom:15px}
.result-name{font-size:1.2em;font-weight:bold;margin-bottom:10px}
.result-phone{color:#667eea;margin-bottom:10px}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>🤖 Chatbot with Reasoning</h1></div>
<div class="search-box">
<input type="text" id="queryInput" placeholder="Enter query..." onkeypress="if(event.key==='Enter')search()">
<button id="searchBtn" onclick="search()">🔍 Search</button>
</div>
<div class="reasoning-container" id="reasoningContainer">
<h2>🤔 Thinking Process</h2>
<div id="reasoningSteps"></div>
</div>
<div class="results-container" id="resultsContainer">
<h2>📊 Results</h2>
<div id="resultsSummary"></div>
<div id="resultsContent"></div>
</div>
</div>
<script>
async function search(){
const query=document.getElementById('queryInput').value.trim();
if(!query)return;
document.getElementById('reasoningContainer').classList.remove('show');
document.getElementById('resultsContainer').classList.remove('show');
document.getElementById('reasoningSteps').innerHTML='';
document.getElementById('reasoningContainer').classList.add('show');
const btn=document.getElementById('searchBtn');
btn.disabled=true;btn.innerHTML='⏳ Searching...';
const es=new EventSource(`http://localhost:8000/api/search?query=${encodeURIComponent(query)}`);
es.onmessage=e=>{
try{
const d=JSON.parse(e.data);
if(d.type==='reasoning_step'){
const div=document.createElement('div');
div.className='reasoning-step';
div.innerHTML=`<strong>Step ${d.step.step}: ${d.step.stage}</strong><br>${d.step.action}`;
document.getElementById('reasoningSteps').appendChild(div);
}else if(d.type==='results'){
document.getElementById('resultsContainer').classList.add('show');
document.getElementById('resultsSummary').innerHTML=`Found: ${d.data.total_found}`;
d.data.results.forEach((r,i)=>{
const card=document.createElement('div');
card.className='result-card';
card.innerHTML=`<div class="result-name">${i+1}. ${r.name}</div><div class="result-phone">📱 ${r.phone}</div><div>Score: ${r.score}</div>`;
document.getElementById('resultsContent').appendChild(card);
});
}else if(d.type==='reasoning_complete'){
btn.disabled=false;btn.innerHTML='🔍 Search';es.close();
}
}catch(err){console.error(err)}
};
es.onerror=()=>{es.close();btn.disabled=false;btn.innerHTML='🔍 Search'};
}
</script>
</body>
</html>
HTML_EOF

echo ""
echo "✅ ALL FILES CREATED!"
echo ""
echo "TO START:"
echo "1. Edit .env with your OpenAI key"
echo "2. Run: ./start.sh"
echo "3. Open: http://localhost:8000/ui"
