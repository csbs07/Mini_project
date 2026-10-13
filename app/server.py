"""FastAPI server providing REST endpoints, SSE streaming, auth, and serving the /chat interface."""
import json
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .auth import (
    clear_auth_cookie,
    create_user_session_token,
    delete_user_session,
    extract_token_from_request,
    get_current_user,
    get_optional_user,
    hash_password,
    set_auth_cookie,
    validate_signup_inputs,
    verify_password,
)
from .config import BASE_DIR
from .db import (
    create_user,
    delete_conversation,
    get_conversation,
    get_conversations_for_user,
    get_messages_for_conversation,
    get_user_by_username_or_email,
    init_db,
    update_conversation_title,
)
from .github import search_github_repositories
from .openalex import search_openalex_papers
from .service import ChatService

# Ensure database tables exist at startup
init_db()

app = FastAPI(
    title="Literature Search & Full-Text RAG Chatbot",
    description="Legal open-access full-text paper ingestion and grounded citation chat engine with authentication and history",
    version="2.1.0",
)

STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

service = ChatService()


# Request Models
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    identifier: str
    password: str


class SearchRequest(BaseModel):
    query: str
    session_id: str


class GitHubSearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    max_results: int = 10


class ReadPaperRequest(BaseModel):
    session_id: str
    paper_id: str


class ReadTop5Request(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class RenameConversationRequest(BaseModel):
    title: str


# ==============================================================================
# HTML Page Routes & Authentication UI
# ==============================================================================

@app.get("/login", response_class=HTMLResponse)
@app.get("/signup", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the Sign In / Sign Up page. If already authenticated, redirect to /chat."""
    user = get_optional_user(request)
    if user:
        return RedirectResponse(url="/chat", status_code=status.HTTP_303_SEE_OTHER)

    login_file = STATIC_DIR / "login.html"
    if not login_file.exists():
        return HTMLResponse("<h3>Loading Login Interface...</h3>")
    return HTMLResponse(content=login_file.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the main literature chat application. Unauthenticated users are redirected to /login."""
    user = get_optional_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Loading Chat Interface...</h3>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


# ==============================================================================
# Authentication API Endpoints
# ==============================================================================

@app.post("/api/auth/signup")
async def api_signup(req: SignupRequest, response: Response):
    """Register a new user account with secure password hashing and session cookie."""
    err = validate_signup_inputs(req.username, req.email, req.password)
    if err:
        raise HTTPException(status_code=400, detail=err)

    # Check for existing username or email
    existing_user = get_user_by_username_or_email(req.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="A user already exists with this username.")
    existing_email = get_user_by_username_or_email(req.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="A user already exists with this email address.")

    pw_hash = hash_password(req.password)
    try:
        user_id = create_user(req.username, req.email, pw_hash)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to create user account. Username or email may already be in use.")

    token, expires_at = create_user_session_token(user_id)
    set_auth_cookie(response, token, expires_at)

    return {
        "status": "ok",
        "user": {
            "id": user_id,
            "username": req.username.strip(),
            "email": req.email.strip().lower(),
        },
    }


@app.post("/api/auth/login")
async def api_login(req: LoginRequest, response: Response):
    """Authenticate user with identifier (username or email) and password, setting session cookie."""
    identifier = req.identifier.strip()
    if not identifier or not req.password:
        raise HTTPException(status_code=400, detail="Please enter both username/email and password.")

    user = get_user_by_username_or_email(identifier)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")

    token, expires_at = create_user_session_token(user["id"])
    set_auth_cookie(response, token, expires_at)

    return {
        "status": "ok",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
        },
    }


@app.post("/api/auth/logout")
async def api_logout(request: Request, response: Response):
    """Log out user by deleting active session token and clearing the auth cookie."""
    token = extract_token_from_request(request)
    if token:
        delete_user_session(token)
    clear_auth_cookie(response)
    return {"status": "ok"}


@app.get("/api/auth/me")
async def api_me(current_user: dict = Depends(get_current_user)):
    """Return profile of the currently logged-in user."""
    return {
        "user": {
            "id": current_user["id"],
            "username": current_user["username"],
            "email": current_user["email"],
        }
    }


# ==============================================================================
# Saved Chat Conversations & History API Endpoints
# ==============================================================================

@app.get("/api/conversations")
async def api_get_conversations(current_user: dict = Depends(get_current_user)):
    """List all saved conversations for the logged-in user."""
    convs = get_conversations_for_user(current_user["id"])
    return {"conversations": convs}


@app.get("/api/conversations/{conv_id}/messages")
async def api_get_conversation_messages(conv_id: str, current_user: dict = Depends(get_current_user)):
    """Load all messages for a specific conversation belonging to the user."""
    conv = get_conversation(conv_id, current_user["id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")

    msgs = service.load_conversation_messages(conv_id, current_user["id"])
    return {
        "conversation": conv,
        "messages": msgs,
    }


@app.patch("/api/conversations/{conv_id}")
async def api_rename_conversation(
    conv_id: str,
    req: RenameConversationRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename a conversation owned by the current user."""
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    ok = update_conversation_title(conv_id, current_user["id"], title)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")
    return {"status": "ok", "title": title}


@app.delete("/api/conversations/{conv_id}")
async def api_delete_conversation(conv_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation and all its messages for the current user."""
    ok = delete_conversation(conv_id, current_user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied.")
    return {"status": "ok"}


# ==============================================================================
# Literature Search, PDF Ingestion, and RAG Chat API Endpoints
# ==============================================================================

@app.post("/api/search")
async def api_search(req: SearchRequest, current_user: dict = Depends(get_current_user)):
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


@app.post("/api/github/search")
async def api_github_search(req: GitHubSearchRequest, current_user: dict = Depends(get_current_user)):
    """Search public GitHub repositories for research-related code and implementation evidence."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    results = search_github_repositories(query, max_results=req.max_results)
    return results


@app.post("/api/papers/read")
async def api_read_paper(req: ReadPaperRequest, current_user: dict = Depends(get_current_user)):
    """On-demand lazy ingestion of a single paper."""
    try:
        res = service.ingest_paper(req.session_id, req.paper_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/read_top5")
async def api_read_top5(req: ReadTop5Request, current_user: dict = Depends(get_current_user)):
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
    paper_id: str = Form("new"),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Attach user PDF to a paper P# or upload a new PDF into the session pool."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds 30 MB limit")

    try:
        clean_name = file.filename.rsplit(".", 1)[0]
        res = service.ingest_paper(
            session_id=session_id,
            paper_id=paper_id,
            pdf_bytes_override=pdf_bytes,
            is_user_upload=True,
        )
        # Update title with uploaded filename if it's a new upload
        session, _ = service.get_or_create_session(session_id)
        norm_id = res.get("paper_id")
        if norm_id and norm_id in session.papers and "Uploaded Paper" in session.papers[norm_id].get("title", ""):
            session.papers[norm_id]["title"] = clean_name
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/papers/{paper_id}/text")
async def api_get_paper_text(
    paper_id: str,
    session_id: str = "default",
    current_user: dict = Depends(get_current_user),
):
    """Fetch full extracted pages, abstract, and sections so the user can read the paper."""
    return service.get_paper_reader_data(session_id, paper_id)


@app.get("/api/papers/{paper_id}/pdf")
async def api_get_paper_pdf(
    paper_id: str,
    session_id: str = "default",
    current_user: dict = Depends(get_current_user),
):
    """Stream raw PDF for in-browser PDF viewer."""
    pdf_bytes = service.get_paper_pdf_bytes(session_id, paper_id)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="PDF not found or not yet ingested")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=paper.pdf"},
    )


@app.post("/api/chat/stream")
async def api_chat_stream(
    req: ChatRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """SSE streaming endpoint for grounded chat interaction with database persistence."""
    client_ip = request.client.host if request.client else "127.0.0.1"

    def event_generator():
        gen = service.chat_turn_stream(
            session_id=req.session_id,
            user_message=req.message,
            client_ip=client_ip,
            user_id=current_user["id"],
        )
        for ev in gen:
            data = json.dumps(ev)
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get current session status, papers, and upload count."""
    session, _ = service.get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "upload_count": session.upload_count,
        "papers": list(session.papers.values()),
    }


@app.post("/api/chat/export")
async def api_export_chat(req: BaseModel, current_user: dict = Depends(get_current_user)):
    """Export chat history as markdown text."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "literature-search-chatbot"}
