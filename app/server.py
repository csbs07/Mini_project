"""FastAPI server providing REST endpoints, SSE streaming, and serving the /chat interface."""
import json

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import BASE_DIR, CHAT_SESSION_COST_CAP_USD
from .openalex import search_openalex_papers
from .service import ChatService

app = FastAPI(
    title="Literature Search & Full-Text RAG Chatbot",
    description="Legal open-access full-text paper ingestion and grounded citation chat engine",
    version="2.0.0",
)

STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

service = ChatService()


class SearchRequest(BaseModel):
    query: str
    session_id: str


class ReadPaperRequest(BaseModel):
    session_id: str
    paper_id: str


class ReadTop5Request(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the single-page vanilla JS literature chat application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Loading Chat Interface...</h3>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.post("/api/search")
async def api_search(req: SearchRequest):
    """Search OpenAlex for top 15 papers for an idea or query and initialize session pool."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    papers = search_openalex_papers(query, per_page=15)
    service.set_session_papers(req.session_id, papers)

    return {
        "session_id": req.session_id,
        "query": query,
        "count": len(papers),
        "papers": papers,
    }


@app.post("/api/papers/read")
async def api_read_paper(req: ReadPaperRequest):
    """On-demand lazy ingestion of a single paper."""
    try:
        res = service.ingest_paper(req.session_id, req.paper_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/read_top5")
async def api_read_top5(req: ReadTop5Request):
    """Read top 5 papers in session pool (capped strictly at 5)."""
    session, _ = service.get_or_create_session(req.session_id)
    paper_keys = list(session.papers.keys())[:5]  # Strictly capped at 5

    results = []
    for pid in paper_keys:
        try:
            r = service.ingest_paper(req.session_id, pid)
            results.append(r)
        except Exception as e:
            results.append({"paper_id": pid, "status": "error", "reason": str(e)})

    return {"processed_count": len(results), "results": results}


@app.post("/api/papers/upload")
async def api_upload_paper(
    session_id: str = Form(...),
    paper_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Attach user PDF to a paper P# (e.g. for paywalled paper)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds 30 MB limit")

    try:
        res = service.ingest_paper(
            session_id=session_id,
            paper_id=paper_id,
            pdf_bytes_override=pdf_bytes,
            is_user_upload=True,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, request: Request):
    """SSE streaming endpoint for grounded chat interaction."""
    client_ip = request.client.host if request.client else "127.0.0.1"

    def event_generator():
        gen = service.chat_turn_stream(
            session_id=req.session_id,
            user_message=req.message,
            client_ip=client_ip,
        )
        for ev in gen:
            data = json.dumps(ev)
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str):
    """Get current session status, papers, upload count, and total cost."""
    session, _ = service.get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "total_cost_usd": round(session.total_cost_usd, 4),
        "cost_cap_usd": CHAT_SESSION_COST_CAP_USD,
        "upload_count": session.upload_count,
        "papers": list(session.papers.values()),
    }


@app.post("/api/chat/export")
async def api_export_chat(req: BaseModel):
    """Export chat history as markdown text."""
    # Handled client-side directly or formatted server-side
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "literature-search-chatbot"}
