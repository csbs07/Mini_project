"""
CLI demo – runs the full pipeline without a browser.
Useful for presentation when network or UI is constrained.
"""
import json
from .retrieval import multi_source_retrieve
from .analysis import run_full_analysis


def main():
    print("=" * 70)
    print("NovaGap – Literature-Grounded Novelty & Gap-Filling Engine")
    print("=" * 70)

    title = (
        "NovaGap: Literature-Grounded Multi-Source Novelty Assessment "
        "with Automatic Research Gap Identification and Idea Enhancement"
    )
    abstract = (
        "We propose a system that accepts a student project idea, retrieves related "
        "academic papers (arXiv) and open-source repositories (GitHub), computes semantic "
        "similarity using Sentence-BERT embeddings, produces an explainable novelty score, "
        "extracts concrete research gaps with evidence, and generates improved project "
        "variants that fill those gaps while remaining feasible for academic timelines."
    )

    print("\n[1] Input idea")
    print(f"Title   : {title[:80]}...")
    print(f"Abstract: {abstract[:100]}...")

    print("\n[2] Multi-source retrieval (arXiv + GitHub + samples)...")
    retrieved = multi_source_retrieve(title, abstract, use_live=True)
    print(f"   Papers retrieved : {len(retrieved['papers'])}")
    print(f"   Repos retrieved  : {len(retrieved['repos'])}")

    print("\n[3] Embedding similarity + novelty scoring + gap analysis...")
    report = run_full_analysis(title, abstract, retrieved["papers"], retrieved["repos"])

    print("\n" + "=" * 70)
    print(f"NOVELTY SCORE      : {report.novelty_score} / 100")
    print(f"SIMILARITY SIGNAL  : {report.similarity_score} / 100")
    print(f"INTERPRETATION     : {report.interpretation}")
    print("=" * 70)

    print("\n--- RESEARCH GAPS ---")
    for i, g in enumerate(report.research_gaps, 1):
        print(f"{i}. [{g.severity.upper()}] {g.description[:120]}...")

    print("\n--- IMPROVED IDEAS (GAP-FILLING) ---")
    for i, idea in enumerate(report.improved_ideas, 1):
        print(f"{i}. {idea.title}")
        print(f"   Feasibility: {idea.estimated_feasibility}")

    print("\n--- TOP SIMILAR PAPER ---")
    if report.top_similar_papers:
        p = report.top_similar_papers[0]
        print(f"{p.title} (sim={p.similarity:.3f})")

    print("\nDemo complete. For interactive UI run: uvicorn app.main:app --reload")
    print("=" * 70)

    # Optional: dump full JSON
    # print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
