import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from app.service import ChatService
from app.openalex import search_openalex_papers

SAMPLE_IDEAS = [
    "Multi-agent reinforcement learning for traffic signal control",
    "Diffusion models for single-cell RNA sequencing data imputation",
    "Retrieval-augmented generation for clinical question answering",
]


def evaluate():
    service = ChatService()
    results = []

    print("=" * 60)
    print("RUNNING END-TO-END EVALUATION ON 3 SAMPLE RESEARCH IDEAS")
    print("=" * 60)

    total_papers_evaluated = 0
    total_fulltext_available = 0
    all_ingestion_times = []
    extraction_failures = []

    for idx, idea in enumerate(SAMPLE_IDEAS, 1):
        sess_id = f"eval_sess_{idx}"
        print(f"\nEvaluating Idea {idx}: '{idea}'")
        t_search_start = time.time()
        papers = search_openalex_papers(idea, per_page=15)
        t_search = time.time() - t_search_start
        print(f"  Retrieved {len(papers)} works from OpenAlex in {t_search:.2f}s")

        service.set_session_papers(sess_id, papers)

        # Ingest top 5 papers
        idea_ft_count = 0
        idea_ingest_times = []

        for p in papers[:5]:
            p_id = p["id"]
            p_label = p["label"]
            t0 = time.time()
            try:
                res = service.ingest_paper(sess_id, p_id)
                dur = time.time() - t0
                status = res.get("status")
                reason = res.get("reason")
                if status == "full_text":
                    idea_ft_count += 1
                    idea_ingest_times.append(dur)
                    all_ingestion_times.append(dur)
                    print(f"    [{p_label}] Full text ingested in {dur:.2f}s ({res.get('page_count')} pages)")
                else:
                    print(f"    [{p_label}] {status}: {reason}")
            except Exception as e:
                dur = time.time() - t0
                extraction_failures.append((p_label, str(e)))
                print(f"    [{p_label}] FAILED in {dur:.2f}s: {e}")

        # Check coverage across top 15 by testing PDF source availability
        top15_ft_count = 0
        for p in papers:
            from app.sources import find_pdf_source
            pdf_url, _, _ = find_pdf_source(p)
            if pdf_url:
                top15_ft_count += 1

        total_papers_evaluated += len(papers)
        total_fulltext_available += top15_ft_count

        # Test Chat turn latency and time-to-first-token
        query = f"What methodology is used in P1?"
        t_turn_start = time.time()
        t_first_token = None
        stream = service.chat_turn_stream(sess_id, query)

        tokens_count = 0
        for ev in stream:
            if ev.get("type") == "delta":
                if t_first_token is None:
                    t_first_token = time.time() - t_turn_start
                tokens_count += 1

        t_total_answer = time.time() - t_turn_start

        results.append({
            "idea": idea,
            "top15_count": len(papers),
            "top15_fulltext_coverage": f"{top15_ft_count}/{len(papers)} ({top15_ft_count/max(1, len(papers))*100:.1f}%)",
            "avg_ingest_seconds": sum(idea_ingest_times) / max(1, len(idea_ingest_times)) if idea_ingest_times else 0.0,
            "time_to_first_token": t_first_token or 0.0,
            "answer_latency": t_total_answer,
        })

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"Idea: {r['idea']}")
        print(f"  Top 15 Full-Text Coverage: {r['top15_fulltext_coverage']}")
        print(f"  Avg Ingestion Time: {r['avg_ingest_seconds']:.2f}s per paper")
        print(f"  Time to First Token: {r['time_to_first_token']:.2f}s")
        print(f"  Total Answer Latency: {r['answer_latency']:.2f}s")

    print("\nOverall Stats:")
    print(f"Total Papers Evaluated: {total_papers_evaluated}")
    print(f"Total with Legal Open-Access Full Text: {total_fulltext_available} ({total_fulltext_available/max(1, total_papers_evaluated)*100:.1f}%)")
    if all_ingestion_times:
        print(f"Mean Ingestion Time: {sum(all_ingestion_times)/len(all_ingestion_times):.2f}s (Median: {sorted(all_ingestion_times)[len(all_ingestion_times)//2]:.2f}s)")
    print(f"Extraction Failures: {len(extraction_failures)}")


if __name__ == "__main__":
    evaluate()
