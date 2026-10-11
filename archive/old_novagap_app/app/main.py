"""
NovaGap FastAPI application – Literature-Grounded Multi-Source Novelty Engine
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import traceback

from .models import ProjectInput, NoveltyReport
from .retrieval import multi_source_retrieve
from .analysis import run_full_analysis

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(
    title="NovaGap – Novelty & Gap-Filling Engine",
    description="Literature-grounded multi-source novelty assessment for student projects",
    version="1.0.0"
)

# Static & templates
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "static"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze", response_model=NoveltyReport)
async def analyze_api(payload: ProjectInput):
    try:
        retrieved = multi_source_retrieve(payload.title, payload.abstract, use_live=True)
        report = run_full_analysis(
            payload.title,
            payload.abstract,
            retrieved["papers"],
            retrieved["repos"]
        )
        return report
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Analysis failed. Try again or use sample data."}
        )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_form(
    request: Request,
    title: str = Form(...),
    abstract: str = Form(...)
):
    try:
        retrieved = multi_source_retrieve(title, abstract, use_live=True)
        report = run_full_analysis(title, abstract, retrieved["papers"], retrieved["repos"])
        return templates.TemplateResponse(
            "report.html",
            {"request": request, "report": report}
        )
    except Exception as e:
        traceback.print_exc()
        return HTMLResponse(
            f"<h2>Error during analysis</h2><pre>{e}</pre><a href='/'>Back</a>",
            status_code=500
        )


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "NovaGap"}
