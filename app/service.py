"""Service layer coordinating sessions, ingestion, hybrid retrieval, tool calls, and streaming."""
import json
import time
from collections.abc import Generator
from typing import Any

from .config import (
    CHAT_MAX_MESSAGE_CHARS,
    CHAT_MAX_UPLOADS_PER_SESSION,
    LLM_CHAT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_RETRIEVAL_CHUNKS,
    MAX_TOOL_CALLS_PER_TURN,
)
from .grounding import (
    SYSTEM_PROMPT,
    sanitize_forbidden_claims,
    validate_and_sanitize_citations,
    validate_and_sanitize_quotes,
)
from .ingestion import (
    chunk_extracted_pages,
    extract_pdf_pages,
    load_ingestion_cache,
    save_ingestion_cache,
)
from .llm import (
    CHAT_TOOLS,
    RateLimiter,
    SessionState,
    get_groq_client,
)
from .openalex import (
    expand_citations_openalex,
    get_openalex_paper,
    normalize_openalex_id,
    search_openalex_papers,
)
from .retrieval import SessionChunkIndex
from .security import (
    SecurityValidationError,
    safe_download_pdf,
)
from .sources import get_candidate_pdf_urls
from .summarizer import map_reduce_summarize_paper


def _call_groq_with_retry(
    groq_client: Any,
    kwargs: dict[str, Any],
    max_retries: int = 3,
) -> tuple[str, Any]:
    """Call Groq completions with retry on 429 rate limit and automatic fallback model."""
    primary_model = kwargs.get("model", LLM_CHAT_MODEL)
    fallback_model = "qwen/qwen3.8-27b"
    models_to_try = [primary_model]
    if fallback_model != primary_model:
        models_to_try.append(fallback_model)

    last_err: Exception | None = None
    for model_name in models_to_try:
        call_kwargs = dict(kwargs)
        call_kwargs["model"] = model_name
        for attempt in range(max_retries):
            try:
                res = groq_client.chat.completions.create(**call_kwargs)
                return model_name, res
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                status_code = getattr(e, "status_code", None)
                if (
                    status_code == 429
                    or "429" in err_str
                    or "rate limit" in err_str
                    or "rate_limit" in err_str
                ):
                    # Groq free-tier TPM rate limits reset in ~0.6-1.5s
                    time.sleep(1.0 * (attempt + 1))
                    continue
                # Non-rate-limit error: do not retry this model
                raise e

    if last_err:
        raise last_err
    raise RuntimeError("Groq call failed with unknown error")


class ChatService:
    """Singleton service managing chat sessions, paper ingestion, and tool orchestration."""

    def __init__(self):
        self.sessions: dict[str, SessionState] = {}
        self.indices: dict[str, SessionChunkIndex] = {}
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)

    def get_or_create_session(self, session_id: str) -> tuple[SessionState, SessionChunkIndex]:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
            self.indices[session_id] = SessionChunkIndex()
        return self.sessions[session_id], self.indices[session_id]

    def set_session_papers(self, session_id: str, papers: list[dict[str, Any]]) -> None:
        """Store papers in session pool and build label mappings (P1, P2, ...)."""
        session, _ = self.get_or_create_session(session_id)
        session.papers.clear()
        session.paper_label_to_id.clear()

        for idx, p in enumerate(papers):
            label = f"P{idx + 1}"
            p["label"] = label
            clean_id = normalize_openalex_id(p["id"])
            session.papers[clean_id] = p
            session.paper_label_to_id[label] = clean_id

    def resolve_paper_id(self, session: SessionState, paper_id_or_label: str) -> str | None:
        """Resolve a label like 'P1' or an ID to the normalized OpenAlex ID."""
        clean = paper_id_or_label.strip()
        if clean.upper() in session.paper_label_to_id:
            return session.paper_label_to_id[clean.upper()]
        norm = normalize_openalex_id(clean)
        if norm in session.papers:
            return norm
        for p_id in session.papers:
            if norm.lower() in p_id.lower():
                return p_id
        return None

    def ingest_paper(
        self,
        session_id: str,
        paper_id: str,
        pdf_bytes_override: bytes | None = None,
        is_user_upload: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest full text of paper into session pool:
        - Check cache in data/fulltext/
        - If not cached and no override, discover legal OA PDF URL and download safely
        - Extract text with PyMuPDF, strip headers/footers, detect sections
        - Check text layer (scanned PDF fallback)
        - Chunk and index in session hybrid retriever
        """
        session, index = self.get_or_create_session(session_id)
        norm_id = self.resolve_paper_id(session, paper_id) or normalize_openalex_id(paper_id)

        paper = session.papers.get(norm_id)
        if not paper:
            # If user uploaded a custom PDF directly
            if pdf_bytes_override:
                label = f"P{len(session.papers) + 1}"
                norm_id = norm_id or f"CUSTOM_{len(session.papers) + 1}"
                paper = {
                    "id": norm_id,
                    "label": label,
                    "title": f"Uploaded Paper ({label})",
                    "year": 2024,
                    "abstract": "Uploaded by user.",
                    "authors": ["User Upload"],
                    "url": "",
                    "full_text_status": "unprocessed",
                    "page_count": None,
                    "sections": [],
                }
                session.papers[norm_id] = paper
                session.paper_label_to_id[label] = norm_id
            else:
                # Try to fetch from OpenAlex if not yet in pool
                paper = get_openalex_paper(norm_id)
                if paper:
                    label = f"P{len(session.papers) + 1}"
                    paper["label"] = label
                    session.papers[norm_id] = paper
                    session.paper_label_to_id[label] = norm_id
                else:
                    return {
                        "paper_id": norm_id,
                        "status": "error",
                        "reason": f"Paper '{paper_id}' not found in pool or OpenAlex.",
                    }

        paper_label = paper.get("label", "P1")

        # Check existing cache (if not user upload override)
        if not pdf_bytes_override:
            cached = load_ingestion_cache(norm_id)
            if cached:
                ext_data, chunks = cached
                metadata = ext_data.get("metadata", {})
                status = metadata.get("status", "full_text")
                reason = metadata.get("reason")
                pages = ext_data.get("pages", [])
                sections = metadata.get("sections", [])
                page_count = len(pages)

                paper["full_text_status"] = status
                paper["full_text_reason"] = reason
                paper["page_count"] = page_count
                paper["sections"] = sections

                if status in ("full_text", "full_text (user upload)") and chunks:
                    index.add_chunks(chunks)

                return {
                    "paper_id": norm_id,
                    "paper_label": paper_label,
                    "status": status,
                    "reason": reason,
                    "page_count": page_count,
                    "sections": sections,
                    "from_cache": True,
                }

        # Handle user upload
        if is_user_upload:
            if session.upload_count >= CHAT_MAX_UPLOADS_PER_SESSION:
                raise SecurityValidationError(
                    f"Session upload limit reached ({CHAT_MAX_UPLOADS_PER_SESSION} uploads maximum)"
                )
            pdf_bytes = pdf_bytes_override
            status_tag = "full_text (user upload)"
        else:
            # Discover all candidate legal OA PDF URLs in priority order
            candidate_urls = get_candidate_pdf_urls(paper)
            if not candidate_urls:
                oa_info = paper.get("open_access", {})
                is_oa = oa_info.get("is_oa", False) if isinstance(oa_info, dict) else False
                reason = "Paywalled (Abstract available)" if not is_oa else "Direct PDF not indexed (Abstract available)"
                paper["full_text_status"] = "abstract_only"
                paper["full_text_reason"] = reason

                # Index abstract into RAG chunks so Q&A still works
                abstract_text = paper.get("abstract") or ""
                if abstract_text:
                    abstract_chunk = {
                        "chunk_id": f"{paper_label}_abstract",
                        "paper_label": paper_label,
                        "paper_id": norm_id,
                        "page_start": 1,
                        "page_end": 1,
                        "section_guess": "abstract",
                        "chunk_index": 0,
                        "text": f"Paper {paper_label}: {paper.get('title')}\nAuthors: {', '.join(paper.get('authors', []))}\nYear: {paper.get('year')}\n\nAbstract:\n{abstract_text}",
                        "is_references": False,
                    }
                    index.add_chunks([abstract_chunk])

                return {
                    "paper_id": norm_id,
                    "paper_label": paper_label,
                    "status": "abstract_only",
                    "reason": reason,
                    "page_count": 0,
                    "sections": [],
                }

            # Try candidate URLs in priority order (handles 403 or non-200 on first source)
            pdf_bytes = None
            last_download_error = None
            for cand_url in candidate_urls:
                try:
                    pdf_bytes = safe_download_pdf(cand_url)
                    if pdf_bytes:
                        status_tag = "full_text"
                        break
                except SecurityValidationError as e:
                    last_download_error = str(e)
                    continue

            if not pdf_bytes:
                friendly_reason = "Paywalled or restricted by publisher"
                if last_download_error and "403" in last_download_error:
                    friendly_reason = "Publisher requires browser (Abstract available)"
                elif last_download_error and "Content-Type" in last_download_error:
                    friendly_reason = "Publisher landing page (Abstract available)"
                elif last_download_error:
                    friendly_reason = last_download_error.replace("Download rejected: ", "")

                paper["full_text_status"] = "abstract_only"
                paper["full_text_reason"] = friendly_reason

                # Index abstract into RAG chunks so Q&A still works
                abstract_text = paper.get("abstract") or ""
                if abstract_text:
                    abstract_chunk = {
                        "chunk_id": f"{paper_label}_abstract",
                        "paper_label": paper_label,
                        "paper_id": norm_id,
                        "page_start": 1,
                        "page_end": 1,
                        "section_guess": "abstract",
                        "chunk_index": 0,
                        "text": f"Paper {paper_label}: {paper.get('title')}\nAuthors: {', '.join(paper.get('authors', []))}\nYear: {paper.get('year')}\n\nAbstract:\n{abstract_text}",
                        "is_references": False,
                    }
                    index.add_chunks([abstract_chunk])

                return {
                    "paper_id": norm_id,
                    "paper_label": paper_label,
                    "status": "abstract_only",
                    "reason": friendly_reason,
                    "page_count": 0,
                    "sections": [],
                }

        # Extract text page by page with PyMuPDF
        pages_data, has_text_layer = extract_pdf_pages(pdf_bytes)
        page_count = len(pages_data)

        # Scanned PDF check: fallback to abstract only
        if not has_text_layer:
            paper["full_text_status"] = "abstract_only"
            paper["full_text_reason"] = "no text layer (scanned PDF)"
            paper["page_count"] = page_count

            abstract_text = paper.get("abstract") or ""
            if abstract_text:
                abstract_chunk = {
                    "chunk_id": f"{paper_label}_abstract",
                    "paper_label": paper_label,
                    "paper_id": norm_id,
                    "page_start": 1,
                    "page_end": 1,
                    "section_guess": "abstract",
                    "chunk_index": 0,
                    "text": f"Paper {paper_label}: {paper.get('title')}\nAuthors: {', '.join(paper.get('authors', []))}\nYear: {paper.get('year')}\n\nAbstract:\n{abstract_text}",
                    "is_references": False,
                }
                index.add_chunks([abstract_chunk])

            save_ingestion_cache(
                norm_id,
                pdf_bytes,
                pages_data,
                [],
                {
                    "status": "abstract_only",
                    "reason": "no text layer",
                    "sections": [],
                    "title": paper.get("title"),
                },
            )
            return {
                "paper_id": norm_id,
                "paper_label": paper_label,
                "status": "abstract_only",
                "reason": "no text layer (scanned PDF)",
                "page_count": page_count,
                "sections": [],
            }

        # Chunk text
        chunks = chunk_extracted_pages(pages_data, paper_label=paper_label, paper_id=norm_id)

        # Collect unique section names
        discovered_sections = []
        for p in pages_data:
            for s in p.get("sections", []):
                if s not in discovered_sections:
                    discovered_sections.append(s)

        # Cache results
        save_ingestion_cache(
            norm_id,
            pdf_bytes,
            pages_data,
            chunks,
            {
                "status": status_tag,
                "reason": None,
                "sections": discovered_sections,
                "title": paper.get("title"),
            },
        )

        if is_user_upload:
            session.upload_count += 1

        paper["full_text_status"] = status_tag
        paper["full_text_reason"] = None
        paper["page_count"] = page_count
        paper["sections"] = discovered_sections

        # Add chunks to session index
        index.add_chunks(chunks)

        return {
            "paper_id": norm_id,
            "paper_label": paper_label,
            "status": status_tag,
            "reason": None,
            "page_count": page_count,
            "sections": discovered_sections,
            "chunk_count": len(chunks),
        }

    def get_paper_reader_data(self, session_id: str, paper_id: str) -> dict[str, Any]:
        """Fetch full extracted pages, abstract, and metadata so the user can read the paper in the UI."""
        session, _ = self.get_or_create_session(session_id)
        norm_id = self.resolve_paper_id(session, paper_id) or normalize_openalex_id(paper_id)
        paper = session.papers.get(norm_id) or {}
        cached = load_ingestion_cache(norm_id)

        from .ingestion import get_cache_paths
        _, pdf_path, _, _ = get_cache_paths(norm_id)
        has_local_pdf = pdf_path.exists()

        direct_pdf_url = paper.get("pdf_url")
        cands = get_candidate_pdf_urls(paper)
        if not direct_pdf_url and cands:
            direct_pdf_url = cands[0]
        landing_url = paper.get("url") or paper.get("doi")

        if cached:
            ext_data, _ = cached
            pages = ext_data.get("pages", [])
            if pages:
                return {
                    "paper_id": norm_id,
                    "label": paper.get("label", ""),
                    "title": paper.get("title", "Paper"),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "doi": paper.get("doi"),
                    "abstract": paper.get("abstract") or "",
                    "pages": pages,
                    "sections": ext_data.get("metadata", {}).get("sections", []),
                    "has_full_text": True,
                    "has_local_pdf": has_local_pdf,
                    "direct_pdf_url": direct_pdf_url,
                    "landing_url": landing_url,
                }

        # Abstract-only or not yet ingested
        return {
            "paper_id": norm_id,
            "label": paper.get("label", ""),
            "title": paper.get("title", "Paper"),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "doi": paper.get("doi"),
            "abstract": paper.get("abstract") or "No abstract indexed for this paper.",
            "pages": [],
            "sections": [],
            "has_full_text": False,
            "has_local_pdf": has_local_pdf,
            "direct_pdf_url": direct_pdf_url,
            "landing_url": landing_url,
            "message": paper.get("full_text_reason") or "Abstract only",
        }

    def get_paper_pdf_bytes(self, session_id: str, paper_id: str) -> bytes | None:
        """Return raw PDF bytes from cache if available."""
        session, _ = self.get_or_create_session(session_id)
        norm_id = self.resolve_paper_id(session, paper_id) or normalize_openalex_id(paper_id)
        from .ingestion import get_cache_paths
        _, pdf_path, _, _ = get_cache_paths(norm_id)
        if pdf_path.exists():
            return pdf_path.read_bytes()
        return None

    def chat_turn_stream(
        self,
        session_id: str,
        user_message: str,
        client_ip: str = "127.0.0.1",
    ) -> Generator[dict[str, Any], None, None]:
        """
        Execute chat turn with streaming SSE events, tool execution loop, and server grounding validation.
        Yields JSON events:
        - {"type": "progress", "message": "..."}
        - {"type": "delta", "content": "..."}
        - {"type": "done", "final_text": "...", "citations": [...], "flags": [...], ...}
        """
        # 1. Rate limiting check
        if not self.rate_limiter.check_rate_limit(client_ip):
            yield {
                "type": "error",
                "detail": "Rate limit exceeded (maximum 60 requests per minute). Please wait.",
            }
            return

        session, index = self.get_or_create_session(session_id)

        # 2. Message length check
        if len(user_message) > CHAT_MAX_MESSAGE_CHARS:
            yield {
                "type": "error",
                "detail": f"Message exceeds limit of {CHAT_MAX_MESSAGE_CHARS} characters.",
            }
            return

        groq_client = get_groq_client()
        retrieved_chunks_this_turn: list[dict[str, Any]] = []

        # Build initial paper context string
        paper_pool_summary = []
        for p_id, p_data in session.papers.items():
            plabel = p_data.get("label", "")
            title = p_data.get("title", "")
            status = p_data.get("full_text_status", "unprocessed")
            paper_pool_summary.append(f"- [{plabel}] {title} (Status: {status})")
        pool_ctx = "\n".join(paper_pool_summary) if paper_pool_summary else "No papers currently in pool."

        # If offline or no GROQ_API_KEY, use high-fidelity deterministic simulator
        if groq_client is None:
            yield {"type": "progress", "message": "Searching passages in paper pool..."}
            # Perform hybrid retrieval on the user's message
            chunks = index.search(user_message, k=min(4, MAX_RETRIEVAL_CHUNKS))
            retrieved_chunks_this_turn.extend(chunks)

            # Check if this is a summarization request
            is_summary = any(kw in user_message.lower() for kw in ["summarize", "summary", "overview"])
            target_p = None
            for p_lbl in session.paper_label_to_id:
                if p_lbl.lower() in user_message.lower():
                    target_p = p_lbl
                    break

            if is_summary and target_p:
                target_id = session.paper_label_to_id[target_p]
                paper_obj = session.papers.get(target_id, {})
                # Ensure ingested
                if paper_obj.get("full_text_status") != "full_text":
                    self.ingest_paper(session_id, target_p)
                p_chunks = [c for c in index.chunks if c.get("paper_label") == target_p]
                retrieved_chunks_this_turn.extend(p_chunks[:6])
                generated_text = map_reduce_summarize_paper(
                    target_p,
                    paper_obj.get("title", ""),
                    p_chunks,
                    client=None,
                )
            elif chunks:
                top_chunk = chunks[0]
                pl = top_chunk["paper_label"]
                pg = top_chunk["page_start"]
                sec = top_chunk["section"]
                preview = top_chunk["text"][:280].strip()
                generated_text = (
                    f"Based on the full text of {pl} in the {sec} section [{pl} p.{pg}]:\n\n"
                    f"{preview}\n\n"
                    f"Would you like more details on the experimental setup or limitations?"
                )
            else:
                generated_text = (
                    "I searched the full text and abstracts in your paper pool, but found no relevant passages "
                    "matching this query. You can try a different search query or upload a PDF for any paywalled paper.\n\n"
                    "Suggested next step: Search for papers on this topic or attach a PDF."
                )

            # Stream simulated tokens
            for word in generated_text.split(" "):
                yield {"type": "delta", "content": word + " "}

            # Server validation
            sanitized_text, cit_flags = validate_and_sanitize_citations(
                generated_text, session.papers, retrieved_chunks_this_turn
            )
            sanitized_text, quote_flags = validate_and_sanitize_quotes(
                sanitized_text, retrieved_chunks_this_turn
            )
            sanitized_text, claim_flags = sanitize_forbidden_claims(sanitized_text)

            all_flags = cit_flags + quote_flags + claim_flags
            session.record_usage(LLM_CHAT_MODEL, 150, 100)

            yield {
                "type": "done",
                "final_text": sanitized_text,
                "retrieved_chunks": retrieved_chunks_this_turn,
                "flags": all_flags,
            }
            return

        # --- Real Groq API Execution Loop with Tools ---
        conv_messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCurrent Paper Pool:\n{pool_ctx}"},
        ]
        # Append session history (last 6 messages)
        conv_messages.extend(session.messages[-6:])
        conv_messages.append({"role": "user", "content": user_message})

        tool_calls_count = 0
        raw_text = ""
        used_model = LLM_CHAT_MODEL

        while tool_calls_count < MAX_TOOL_CALLS_PER_TURN:
            try:
                used_model, response = _call_groq_with_retry(
                    groq_client,
                    {
                        "model": LLM_CHAT_MODEL,
                        "messages": conv_messages,
                        "tools": CHAT_TOOLS,
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": LLM_MAX_TOKENS,
                    },
                )
            except Exception as e:
                yield {"type": "error", "detail": f"Groq API error: {e}"}
                return

            choice = response.choices[0]
            msg = choice.message
            prompt_toks = response.usage.prompt_tokens if response.usage else 200
            comp_toks = response.usage.completion_tokens if response.usage else 50
            session.record_usage(used_model, prompt_toks, comp_toks)

            if not msg.tool_calls:
                # No more tools needed, proceed to text response
                raw_text = msg.content or ""
                break

            # Execute tool calls
            conv_messages.append(msg)
            for tc in msg.tool_calls:
                tool_calls_count += 1
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")

                if fn_name == "read_paper":
                    p_id = fn_args.get("paper_id", "")
                    yield {"type": "progress", "message": f"Reading {p_id}..."}
                    result = self.ingest_paper(session_id, p_id)
                elif fn_name == "search_paper_text":
                    q = fn_args.get("query", "")
                    p_ids = fn_args.get("paper_ids")
                    k_val = min(fn_args.get("k", 3), MAX_RETRIEVAL_CHUNKS)
                    yield {"type": "progress", "message": f"Searching passages for '{q[:30]}'..."}
                    chunks = index.search(q, paper_ids=p_ids, k=k_val)
                    retrieved_chunks_this_turn.extend(chunks)
                    result = {
                        "results_count": len(chunks),
                        "chunks": [
                            {
                                "paper": c["paper_label"],
                                "page": c["page_start"],
                                "section": c["section"],
                                "text": c["text"][:500] if len(c["text"]) > 500 else c["text"],
                            }
                            for c in chunks
                        ],
                    }
                elif fn_name == "search_papers":
                    q = fn_args.get("query", "")
                    yield {"type": "progress", "message": f"Searching papers for '{q[:30]}'..."}
                    found = search_openalex_papers(q, per_page=10)
                    self.set_session_papers(session_id, found)
                    result = {
                        "count": len(found),
                        "papers": [{"label": p["label"], "title": p["title"]} for p in found],
                    }
                elif fn_name == "get_paper":
                    p_id = fn_args.get("paper_id", "")
                    p_data = session.papers.get(self.resolve_paper_id(session, p_id) or "")
                    result = p_data or {"error": "Paper not found"}
                elif fn_name == "expand_citations":
                    p_id = fn_args.get("paper_id", "")
                    norm_id = self.resolve_paper_id(session, p_id) or p_id
                    refs = expand_citations_openalex(norm_id, limit=5)
                    result = {"referenced_works": [r["title"] for r in refs]}
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                conv_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

                if tool_calls_count >= MAX_TOOL_CALLS_PER_TURN:
                    break

        # Streaming response: if model already generated text in the loop, stream it directly
        if raw_text:
            chunk_size = 20
            for i in range(0, len(raw_text), chunk_size):
                yield {"type": "delta", "content": raw_text[i : i + chunk_size]}
        else:
            # Model executed tool calls and needs explicit final text synthesis
            synth_messages = list(conv_messages)
            synth_messages.append({
                "role": "user",
                "content": (
                    "Please synthesize your final response for the researcher based strictly on the retrieved "
                    "information above. Cite passages using [P# p.N] format where applicable. Provide your answer directly in text."
                ),
            })
            try:
                _used_final, stream = _call_groq_with_retry(
                    groq_client,
                    {
                        "model": LLM_CHAT_MODEL,
                        "messages": synth_messages,
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": LLM_MAX_TOKENS,
                        "stream": True,
                    },
                )
                accumulated_chunks = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        c_text = chunk.choices[0].delta.content
                        accumulated_chunks.append(c_text)
                        yield {"type": "delta", "content": c_text}
                raw_text = "".join(accumulated_chunks)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    raw_text = "Rate limit reached on Groq free tier. Please retry in a few moments."
                else:
                    raw_text = f"Error generating final response: {e}"
                yield {"type": "delta", "content": raw_text}

        # Server-enforced validation
        sanitized_text, cit_flags = validate_and_sanitize_citations(
            raw_text, session.papers, retrieved_chunks_this_turn
        )
        sanitized_text, quote_flags = validate_and_sanitize_quotes(
            sanitized_text, retrieved_chunks_this_turn
        )
        sanitized_text, claim_flags = sanitize_forbidden_claims(sanitized_text)

        all_flags = cit_flags + quote_flags + claim_flags

        # Update session history (safe without dumping large PDFs or raw chunks)
        session.messages.append({"role": "user", "content": user_message})
        session.messages.append({"role": "assistant", "content": sanitized_text})

        yield {
            "type": "done",
            "final_text": sanitized_text,
            "retrieved_chunks": retrieved_chunks_this_turn,
            "flags": all_flags,
        }
