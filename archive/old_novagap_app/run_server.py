"""
NovaGap Localhost Server Runner
--------------------------------
Automatically detects if full requirements (FastAPI/Uvicorn) are available.
- If available: Runs full FastAPI app with Sentence-Transformers embeddings.
- If NOT available: Falls back to a zero-dependency Python HTTP server using standard library.
"""

import sys
import os
import urllib.parse
import json
import re
import math
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

SAMPLE_PAPERS = [
    {
        "title": "Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers",
        "abstract": "We perform the first head-to-head comparison between expert NLP researchers and an LLM ideation agent. LLM-generated ideas are judged as more novel than human expert ideas while being slightly weaker on feasibility.",
        "year": 2024,
        "url": "https://arxiv.org/abs/2409.04109",
        "source": "sample"
    },
    {
        "title": "Literature-Grounded Novelty Assessment of Scientific Ideas",
        "abstract": "We propose the Idea Novelty Checker, an LLM-based RAG framework that uses two-stage retrieve-then-rerank and expert-labeled examples to evaluate novelty of scientific ideas with literature grounding.",
        "year": 2025,
        "url": "https://arxiv.org/abs/2506.22026",
        "source": "sample"
    },
    {
        "title": "Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and LLM Knowledge",
        "abstract": "We integrate human peer-review knowledge and LLM summaries of methodology sections to train models that predict method novelty of academic papers.",
        "year": 2025,
        "url": "https://arxiv.org/abs/2507.11330",
        "source": "sample"
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "RAG combines a pre-trained retriever with a pre-trained generator to improve knowledge-intensive tasks and reduce hallucination.",
        "year": 2020,
        "url": "https://arxiv.org/abs/2005.11401",
        "source": "sample"
    },
    {
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "abstract": "We present Sentence-BERT, a modification of BERT that uses siamese networks to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.",
        "year": 2019,
        "url": "https://arxiv.org/abs/1908.10084",
        "source": "sample"
    },
    {
        "title": "SciMON: Scientific Inspiration Machines Optimized for Novelty",
        "abstract": "SciMON generates novel scientific ideas by retrieving and combining concepts from literature while optimizing for novelty.",
        "year": 2024,
        "url": "https://arxiv.org/abs/2305.14647",
        "source": "sample"
    }
]

SAMPLE_REPOS = [
    {
        "title": "awesome-ai-auto-research",
        "description": "A curated list of AI systems and papers for automated research idea generation, literature review, and scientific discovery agents.",
        "url": "https://github.com/worldbench/awesome-ai-auto-research",
        "stars": 1200,
        "language": "Markdown",
        "source": "sample"
    },
    {
        "title": "idea_novelty_checker",
        "description": "Implementation related to literature-grounded novelty assessment of scientific ideas using RAG and re-ranking.",
        "url": "https://github.com/simra-shahid/idea_novelty_checker",
        "stars": 85,
        "language": "Python",
        "source": "sample"
    },
    {
        "title": "sentence-transformers",
        "description": "State-of-the-art sentence, text and image embeddings. Used for semantic similarity and retrieval.",
        "url": "https://github.com/UKPLab/sentence-transformers",
        "stars": 15000,
        "language": "Python",
        "source": "sample"
    },
    {
        "title": "langchain",
        "description": "Framework for developing applications powered by language models, including RAG pipelines for literature.",
        "url": "https://github.com/langchain-ai/langchain",
        "stars": 90000,
        "language": "Python",
        "source": "sample"
    }
]

def calculate_word_overlap_similarity(text1, text2):
    def tokenize(txt):
        return set(re.findall(r'\w+', txt.lower()))
    w1 = tokenize(text1)
    w2 = tokenize(text2)
    if not w1 or not w2:
        return 0.0
    intersection = len(w1 & w2)
    union = len(w1 | w2)
    jaccard = intersection / union if union > 0 else 0.0
    overlap_coefficient = intersection / min(len(w1), len(w2)) if min(len(w1), len(w2)) > 0 else 0.0
    return round(0.4 * jaccard + 0.6 * overlap_coefficient, 3)

def run_pure_python_analysis(title, abstract):
    query_text = f"{title} {abstract}"
    
    ranked_papers = []
    for p in SAMPLE_PAPERS:
        sim = calculate_word_overlap_similarity(query_text, f"{p['title']} {p['abstract']}")
        ranked_papers.append({
            "title": p["title"],
            "abstract_or_description": p["abstract"],
            "url": p["url"],
            "year": p["year"],
            "source": p["source"],
            "similarity": sim,
            "extra": {}
        })
    ranked_papers.sort(key=lambda x: x["similarity"], reverse=True)

    ranked_repos = []
    for r in SAMPLE_REPOS:
        sim = calculate_word_overlap_similarity(query_text, f"{r['title']} {r['description']}")
        ranked_repos.append({
            "title": r["title"],
            "abstract_or_description": r["description"],
            "url": r["url"],
            "source": r["source"],
            "similarity": sim,
            "extra": {"stars": r.get("stars"), "language": r.get("language")}
        })
    ranked_repos.sort(key=lambda x: x["similarity"], reverse=True)

    top_sims = [p["similarity"] for p in ranked_papers[:3]] + [r["similarity"] for r in ranked_repos[:2]]
    avg_sim = sum(top_sims) / len(top_sims) if top_sims else 0.2
    
    novelty = max(0.0, min(100.0, (1.0 - avg_sim) * 100))
    similarity_score = round(100.0 - novelty, 1)
    novelty = round(novelty, 1)

    if novelty >= 70:
        interpretation = "High novelty – the project idea presents unique elements compared to existing literature and open-source repos."
    elif novelty >= 45:
        interpretation = "Moderate novelty – closely related work exists; clear opportunities remain for differentiation."
    else:
        interpretation = "Low novelty – strong similarity with existing papers or repositories. Consider filling specific missing gaps."

    gaps = [
        {
            "severity": "high",
            "description": "Joint multi-source retrieval combining academic literature + code repositories is rarely implemented in student project evaluators.",
            "evidence": ["Literature review of AI research agents (2024-2026)."]
        },
        {
            "severity": "medium",
            "description": "Existing evaluation tools stop at giving a numerical score without recommending actionable gap-filling project improvements.",
            "evidence": ["Analysis of prior novelty assessment frameworks."]
        },
        {
            "severity": "medium",
            "description": "Prior literature requires explicit explainability: students need to see exact paper citations and similarity breakdowns.",
            "evidence": ["Shahid et al. (2025) literature-grounded evaluation guidelines."]
        }
    ]

    improved_ideas = [
        {
            "title": "Multi-Source Novelty Evaluator with Interactive Gap Mapping",
            "description": "Extend basic novelty search by integrating arXiv API + GitHub REST API with interactive visual heatmaps comparing problem statements against retrieved abstracts.",
            "how_it_fills_gaps": [
                "Combines paper literature and open-source code.",
                "Provides actionable feedback instead of a static score."
            ],
            "estimated_feasibility": "high"
        },
        {
            "title": "Facet-Based Idea Novelty & Feasibility Co-Scorer",
            "description": "Decompose student proposals into Problem, Method, Dataset, and Evaluation facets, scoring each against literature and rating student timeline feasibility.",
            "how_it_fills_gaps": [
                "Improves explainability for students and faculty guides.",
                "Balances novelty with realistic student feasibility."
            ],
            "estimated_feasibility": "medium"
        }
    ]

    lit_summary = "### Key Related Literature & Repositories\n"
    for i, p in enumerate(ranked_papers[:4], 1):
        lit_summary += f"{i}. {p['title']} ({p['year']}) – Similarity: {p['similarity']:.2f}\n"
    for i, r in enumerate(ranked_repos[:3], 1):
        lit_summary += f"{i}. Repository: {r['title']} – Similarity: {r['similarity']:.2f}\n"

    refs = [f"{p['title']} – {p['url']}" for p in ranked_papers[:5]]

    return {
        "input_title": title,
        "novelty_score": novelty,
        "similarity_score": similarity_score,
        "interpretation": interpretation,
        "top_similar_papers": ranked_papers[:5],
        "top_similar_repos": ranked_repos[:4],
        "research_gaps": gaps,
        "improved_ideas": improved_ideas,
        "literature_survey_summary": lit_summary,
        "references": refs,
        "method_notes": "Zero-dependency pure Python search mode (Jaccard + Overlap keyword indexing). No external requirements required."
    }

class NovaGapHTTPRequestHandler(BaseHTTPRequestHandler):
    def send_html(self, content: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content.encode('utf-8'))))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def send_json(self, data: dict, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            idx_file = STATIC_DIR / "index.html"
            if idx_file.exists():
                self.send_html(idx_file.read_text(encoding="utf-8"))
            else:
                self.send_html("<h1>NovaGap Server Running</h1><p>Index page not found.</p>")
        elif path == "/demo":
            demo_file = STATIC_DIR / "demo.html"
            if demo_file.exists():
                self.send_html(demo_file.read_text(encoding="utf-8"))
            else:
                self.send_html("<h1>Demo Page</h1>")
        elif path == "/health":
            self.send_json({"status": "ok", "service": "NovaGap-ZeroRequirements"})
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            target_file = STATIC_DIR / rel_path
            if target_file.exists() and target_file.is_file():
                content = target_file.read_bytes()
                mime = "text/html"
                if target_file.suffix == ".css": mime = "text/css"
                elif target_file.suffix == ".js": mime = "application/javascript"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "File Not Found")
        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        if parsed.path == "/analyze":
            form_params = urllib.parse.parse_qs(post_data)
            title = form_params.get("title", [""])[0]
            abstract = form_params.get("abstract", [""])[0]

            report = run_pure_python_analysis(title, abstract)
            html = render_report_html(report)
            self.send_html(html)
        elif parsed.path == "/api/analyze":
            try:
                payload = json.loads(post_data)
                title = payload.get("title", "")
                abstract = payload.get("abstract", "")
                report = run_pure_python_analysis(title, abstract)
                self.send_json(report)
            except Exception as e:
                self.send_json({"error": str(e)}, status=400)
        else:
            self.send_error(404, "Endpoint Not Found")

def render_report_html(report: dict) -> str:
    template_path = STATIC_DIR / "report.html"
    if not template_path.exists():
        return f"<h2>Analysis Complete</h2><pre>{json.dumps(report, indent=2)}</pre>"
    
    html = template_path.read_text(encoding="utf-8")
    
    # Simple template engine replacement for standalone mode
    html = html.replace("{{ report.input_title }}", html_escape(report["input_title"]))
    html = html.replace("{{ report.novelty_score }}", str(report["novelty_score"]))
    html = html.replace("{{ report.similarity_score }}", str(report["similarity_score"]))
    html = html.replace("{{ report.interpretation }}", html_escape(report["interpretation"]))
    html = html.replace("{{ report.literature_survey_summary }}", html_escape(report["literature_survey_summary"]))
    html = html.replace("{{ report.method_notes }}", html_escape(report["method_notes"]))

    # Render Gaps
    gaps_html = ""
    for g in report["research_gaps"]:
        sev = g["severity"]
        ev_items = "".join([f'<li style="color:var(--muted);font-size:0.9rem">{html_escape(e)}</li>' for e in g["evidence"]])
        gaps_html += f'''
        <div class="item gap {sev}">
          <span class="tag {sev}">{sev.upper()}</span>
          {html_escape(g["description"])}
          <ul>{ev_items}</ul>
        </div>
        '''
    html = re.sub(r'\{%\s*for g in report\.research_gaps\s*%\}.*?\{%\s*endfor\s*%\}', gaps_html, html, flags=re.DOTALL)

    # Render Improved Ideas
    ideas_html = ""
    for idea in report["improved_ideas"]:
        feas = idea["estimated_feasibility"]
        fills = "".join([f'<li>{html_escape(h)}</li>' for h in idea["how_it_fills_gaps"]])
        ideas_html += f'''
        <div class="item idea">
          <strong>{html_escape(idea["title"])}</strong>
          <span class="tag {feas}">Feasibility: {feas}</span>
          <p style="margin-top:0.5rem">{html_escape(idea["description"])}</p>
          <p style="margin-top:0.5rem;color:var(--muted)"><em>How it fills gaps:</em></p>
          <ul>{fills}</ul>
        </div>
        '''
    html = re.sub(r'\{%\s*for idea in report\.improved_ideas\s*%\}.*?\{%\s*endfor\s*%\}', ideas_html, html, flags=re.DOTALL)

    # Render Papers
    papers_html = ""
    for p in report["top_similar_papers"]:
        sim_str = f"{p['similarity']:.2f}"
        year_str = f'<span class="tag">{p["year"]}</span>' if p.get("year") else ''
        url_str = f'<a href="{p["url"]}" target="_blank">Open</a>' if p.get("url") else ''
        papers_html += f'''
        <div class="item">
          <span class="sim">sim {sim_str}</span>
          <strong>{html_escape(p["title"])}</strong>
          {year_str}
          <span class="tag">{p["source"]}</span>
          <p style="margin-top:0.4rem;font-size:0.92rem;color:var(--muted)">{html_escape(p["abstract_or_description"][:280])}...</p>
          {url_str}
        </div>
        '''
    html = re.sub(r'\{%\s*for p in report\.top_similar_papers\s*%\}.*?\{%\s*endfor\s*%\}', papers_html, html, flags=re.DOTALL)

    # Render Repos
    repos_html = ""
    for r in report["top_similar_repos"]:
        sim_str = f"{r['similarity']:.2f}"
        stars = r.get("extra", {}).get("stars")
        star_str = f'<span class="tag">★ {stars}</span>' if stars else ''
        url_str = f'<a href="{r["url"]}" target="_blank">Open</a>' if r.get("url") else ''
        repos_html += f'''
        <div class="item">
          <span class="sim">sim {sim_str}</span>
          <strong>{html_escape(r["title"])}</strong>
          {star_str}
          <span class="tag">{r["source"]}</span>
          <p style="margin-top:0.4rem;font-size:0.92rem;color:var(--muted)">{html_escape(r["abstract_or_description"][:220])}...</p>
          {url_str}
        </div>
        '''
    html = re.sub(r'\{%\s*for r in report\.top_similar_repos\s*%\}.*?\{%\s*endfor\s*%\}', repos_html, html, flags=re.DOTALL)

    # Render References
    refs_html = ""
    for ref in report["references"]:
        refs_html += f'<li style="margin-bottom:0.35rem">{html_escape(ref)}</li>'
    html = re.sub(r'\{%\s*for ref in report\.references\s*%\}.*?\{%\s*endfor\s*%\}', refs_html, html, flags=re.DOTALL)

    return html

def html_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

import socket

def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def main():
    host = "127.0.0.1"
    ports_to_try = [8000, 8080, 8050, 8888, 8001, 8081, 8090, 9000]

    # Check if uvicorn & fastapi are available
    has_requirements = False
    try:
        import uvicorn
        import fastapi
        has_requirements = True
    except ImportError:
        pass

    target_port = None
    for p in ports_to_try:
        if is_port_available(host, p):
            target_port = p
            break

    if target_port is None:
        print("[!] All attempted ports (8000, 8080, 8050, 8888) are in use.")
        return

    if has_requirements:
        import uvicorn
        print(f"[+] Full requirements detected (FastAPI + Uvicorn).")
        print(f"[*] Starting NovaGap server on http://{host}:{target_port}...")
        uvicorn.run("app.main:app", host=host, port=target_port, reload=False)
    else:
        print("------------------------------------------------------------------")
        print("[-] Running in Standalone Zero-Dependency Mode")
        print(f"[*] Starting NovaGap localhost server on http://{host}:{target_port}...")
        print(f"   Open http://{host}:{target_port} in your web browser.")
        print("   Press Ctrl+C to stop the server.")
        print("------------------------------------------------------------------")
        server_address = (host, target_port)
        httpd = HTTPServer(server_address, NovaGapHTTPRequestHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping NovaGap server.")
            httpd.server_close()

if __name__ == "__main__":
    main()
