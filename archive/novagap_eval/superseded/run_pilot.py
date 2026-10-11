"""
run_pilot.py — NovaGap-Lite Verification Pilot
Runs the full experiment pipeline end-to-end:
  1. Ingest CS papers from OpenAlex (free, no API key)
  2. Build temporal split at T=2018 (train < 2018, eval 2018-2021)
  3. Sample degree-matched positive/negative pairs
  4. Score all baselines + NovaGap-Lite gated novelty
  5. Report AUROC / Avg Precision / P@50 for all methods
  6. RQ2: citation association analysis
  7. RQ3: gate ablation (mirage reduction)

Run from the project root:
  python research/run_pilot.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path
from dataset_builder import fetch_openalex_cs, build_temporal_split
from evaluator import TemporalEvaluator

# -----------------------------------------------------------------------
# CONFIG — adjust for a quick pilot vs. full experiment
# -----------------------------------------------------------------------
CONFIG = {
    "from_year":     2013,
    "to_year":       2021,
    "per_page":      200,
    "max_pages":     8,          # ~1,600 papers — fast pilot; increase to 25 for full run
    "cutoff_T":      2018,
    "eval_window":   3,          # evaluate 2018-2021
    "n_pos":         400,
    "n_neg":         400,
    "max_path_len":  3,
    "min_aa":        0.05,
    "cache_file":    Path(__file__).parent / "pilot_papers_cache.json",
}


def load_or_fetch(cfg: dict) -> list:
    """Cache the fetched papers so re-runs are instant."""
    if cfg["cache_file"].exists():
        print(f"[Cache] Loading papers from {cfg['cache_file']}")
        with open(cfg["cache_file"]) as f:
            return json.load(f)
    print("[OpenAlex] Fetching papers (this takes ~30-60 seconds)...")
    papers = fetch_openalex_cs(
        from_year=cfg["from_year"],
        to_year=cfg["to_year"],
        per_page=cfg["per_page"],
        max_pages=cfg["max_pages"],
    )
    with open(cfg["cache_file"], "w") as f:
        json.dump(papers, f)
    print(f"[Cache] Saved {len(papers)} papers to {cfg['cache_file']}")
    return papers


def banner(title: str):
    line = "═" * 60
    print(f"\n{line}\n  {title}\n{line}")


def main():
    banner("NovaGap-Lite — Hypothesis Verification Pilot")

    # Step 1: Data
    papers = load_or_fetch(CONFIG)
    print(f"\n✓ Total papers loaded: {len(papers)}")

    # Concept frequency stats
    from collections import Counter
    all_concepts = [c for p in papers for c in p["concepts"]]
    top_concepts = Counter(all_concepts).most_common(10)
    print(f"✓ Total concept tokens: {len(all_concepts)}")
    print(f"✓ Top-10 concepts: {[f'{c}({n})' for c, n in top_concepts]}")

    # Step 2: Temporal split
    print(f"\n⚙  Building temporal split (cutoff T={CONFIG['cutoff_T']})...")
    G_train, test_co, test_cites = build_temporal_split(
        papers,
        cutoff_T=CONFIG["cutoff_T"],
        eval_window_years=CONFIG["eval_window"],
    )
    print(f"✓ G_train: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
    print(f"✓ Distinct pairs in test window: {len(test_co)}")
    pos_pairs = sum(1 for v in test_co.values() if v >= 2)
    print(f"✓ Candidate positives (count≥2, absent in train): {pos_pairs}")

    # Step 3: Sample evaluation pairs
    print(f"\n⚙  Sampling {CONFIG['n_pos']} positive + {CONFIG['n_neg']} negative pairs...")
    evaluator = TemporalEvaluator(G_train, test_co, test_cites)
    dataset   = evaluator.sample_pairs(CONFIG["n_pos"], CONFIG["n_neg"])
    n_pos = sum(1 for d in dataset if d["label"] == 1)
    n_neg = sum(1 for d in dataset if d["label"] == 0)
    print(f"✓ Dataset: {n_pos} positives, {n_neg} negatives")

    # Step 4: Score
    print("\n⚙  Computing scores for all methods...")
    scored = evaluator.score_dataset(
        dataset,
        max_path_len=CONFIG["max_path_len"],
        min_aa=CONFIG["min_aa"],
    )
    plausible_count = sum(1 for d in scored if d["is_plausible"])
    print(f"✓ Pairs passing plausibility gate: {plausible_count}/{len(scored)} "
          f"({100*plausible_count/len(scored):.1f}%)")

    # -----------------------------------------------------------------------
    # RQ1 — Link prediction accuracy
    # -----------------------------------------------------------------------
    banner("RQ1: Can the gated score predict which concept pairs materialize?")
    rq1 = evaluator.evaluate(scored)

    col_w = 26
    print(f"\n{'Method':<{col_w}} │ {'AUROC':>7} │ {'Avg Prec':>9} │ {'P@50':>6}")
    print("─" * (col_w + 32))
    for method, m in rq1.items():
        marker = " ◄" if method == "NovaGap-Lite (Gated)" else ""
        print(f"{method:<{col_w}} │ {m['AUROC']:>7.4f} │ {m['Avg_Precision']:>9.4f} │ {m['P@50']:>6.4f}{marker}")

    novagap_auroc = rq1["NovaGap-Lite (Gated)"]["AUROC"]
    best_baseline = max(
        {k: v for k, v in rq1.items() if "NovaGap" not in k}.items(),
        key=lambda x: x[1]["AUROC"]
    )
    delta = novagap_auroc - best_baseline[1]["AUROC"]
    print(f"\n→ Best baseline: {best_baseline[0]} (AUROC {best_baseline[1]['AUROC']:.4f})")
    print(f"→ NovaGap-Lite delta: {delta:+.4f} vs. best baseline")
    if delta > 0:
        print("✓ Hypothesis SUPPORTED: gated score outperforms all baselines on RQ1.")
    else:
        print("✗ Hypothesis NOT supported on this sample — check gate thresholds & data volume.")

    # -----------------------------------------------------------------------
    # RQ2 — Citation impact association
    # -----------------------------------------------------------------------
    banner("RQ2: Do high-novelty pairs attract more citations?")
    rq2 = evaluator.rq2_citation_analysis(scored)
    for k, v in rq2.items():
        print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # RQ3 — Gate ablation (mirage reduction)
    # -----------------------------------------------------------------------
    banner("RQ3: Does the plausibility gate reduce mirage gaps?")
    rq3 = evaluator.rq3_gate_ablation(scored)
    for k, v in rq3.items():
        print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Summary verdict
    # -----------------------------------------------------------------------
    banner("PILOT VERDICT")
    pass_count = 0
    if delta > 0:
        print("  ✓ RQ1 PASSED: Gated novelty beats all baselines on link prediction.")
        pass_count += 1
    else:
        print("  ✗ RQ1 FAILED on this pilot sample.")

    if rq2.get("spearman_rho", 0) > 0:
        print("  ✓ RQ2 SUPPORTED: High-novelty pairs have higher citation impact.")
        pass_count += 1
    else:
        print("  ✗ RQ2 NOT supported (null result — still publishable if reported honestly).")

    if rq3.get("gate_improves_precision"):
        print("  ✓ RQ3 SUPPORTED: Plausibility gate reduces mirage candidates.")
        pass_count += 1
    else:
        print("  ✗ RQ3 NOT supported — recalibrate max_path_len / min_aa in CONFIG.")

    print(f"\n  Score: {pass_count}/3 research questions supported.")
    if pass_count >= 2:
        print("  → IDEA VERIFIED: Core hypothesis holds. Proceed to full experiment.")
    elif pass_count == 1:
        print("  → PARTIAL: One RQ holds. Review data volume (increase max_pages) and gate thresholds.")
    else:
        print("  → REVISIT: No RQ supported on this pilot. Check API data quality and concept normalization.")


if __name__ == "__main__":
    main()
