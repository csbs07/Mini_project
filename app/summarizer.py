"""Map-Reduce summarizer for whole paper summaries without giant context overflows."""
from typing import Any

import groq

from .config import LLM_CHAT_MODEL, LLM_TEMPERATURE


def map_reduce_summarize_paper(
    paper_label: str,
    paper_title: str,
    chunks: list[dict[str, Any]],
    client: groq.Groq | None = None,
    model: str = LLM_CHAT_MODEL,
) -> str:
    """
    Summarize a whole paper using Map-Reduce over section chunks:
    1. Filter out references chunks.
    2. Group chunks by section.
    3. Map: generate a concise summary per section (capped tokens).
    4. Reduce: synthesize the section summaries into an integrated executive summary with page citations.
    """
    valid_chunks = [c for c in chunks if not c.get("is_references", False)]
    if not valid_chunks:
        return f"No full-text content available to summarize for {paper_label}."

    # Group by section
    sections: dict[str, list[dict[str, Any]]] = {}
    for chk in valid_chunks:
        sec = chk.get("section_guess") or "General"
        sections.setdefault(sec, []).append(chk)

    # If client is not available or mocked, synthesize directly from chunk leaders
    if client is None:
        sec_summaries = []
        for sec_name, sec_chunks in sections.items():
            first_chunk = sec_chunks[0]
            p_start = first_chunk["page_start"]
            preview = first_chunk["text"][:250].strip()
            sec_summaries.append(f"**{sec_name}** [{paper_label} p.{p_start}]: {preview}...")
        return (
            f"### Summary of {paper_label}: {paper_title}\n\n"
            + "\n\n".join(sec_summaries)
            + f"\n\n*Suggested next step: Inquire about specific methodology or dataset details in {paper_label}.*"
        )

    # 1. MAP STAGE: summarize each section
    section_summaries = []
    for sec_name, sec_chunks in sections.items():
        # Combine section text (cap to ~2,000 words to avoid overflows)
        combined_text = " ".join(c["text"] for c in sec_chunks)[:6000]
        p_start = sec_chunks[0]["page_start"]

        prompt = (
            f"Summarize the following '{sec_name}' section of academic paper '{paper_title}' "
            f"in 2-3 concise sentences. Focus strictly on facts and specifics. "
            f"Cite as [{paper_label} p.{p_start}].\n\nSection Text:\n{combined_text}"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise scientific research summarizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=250,
            )
            summary_text = resp.choices[0].message.content.strip()
            section_summaries.append(f"### {sec_name}\n{summary_text}")
        except Exception:
            # Fallback on map error
            section_summaries.append(
                f"### {sec_name} [{paper_label} p.{p_start}]\n{sec_chunks[0]['text'][:200]}..."
            )

    # 2. REDUCE STAGE: synthesize overall paper summary
    all_mapped_text = "\n\n".join(section_summaries)
    reduce_prompt = (
        f"Synthesize the following section summaries of paper '{paper_title}' into a coherent, "
        f"structured whole-paper executive summary. Maintain the citations [{paper_label} p.N].\n\n"
        f"Structure your response with:\n"
        f"- **Problem & Objective**\n"
        f"- **Methodology & Architecture**\n"
        f"- **Key Results & Findings**\n"
        f"- **Limitations**\n\n"
        f"Section Summaries:\n{all_mapped_text}\n\n"
        f"Conclude with one suggested next step."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert research assistant synthesizing paper summaries. "
                        "Follow strict grounding and preserve page citations [P# p.N]."
                    ),
                },
                {"role": "user", "content": reduce_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"## Summary of {paper_label}: {paper_title}\n\n" + all_mapped_text
