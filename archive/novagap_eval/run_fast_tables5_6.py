import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import math, json, random, time, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats as stats
import networkx as nx
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Fast AUROC via Mann-Whitney U
def fast_auroc(y_true, scores):
    pos = (np.asarray(y_true) == 1)
    n_pos = int(np.sum(pos))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(scores)
    u = np.sum(ranks[pos]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Import functions from audit_pilot
import audit_pilot

# Precompute embeddings cache across seeds
concepts_cache = {}

def get_seed_d_aucs(seed):
    seed_file = f"research/openalex_sample_seed{seed}.json"
    with open(seed_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Within-year percentile ranks
    papers_by_year = defaultdict(list)
    for p in papers:
        papers_by_year[p["year"]].append(p)
    for y, p_list in papers_by_year.items():
        cites = [p.get("citations", 0) for p in p_list]
        ranks = stats.rankdata(cites, method="average") / len(cites) * 100.0
        for p, r in zip(p_list, ranks):
            p["pct_rank"] = float(r)

    extractor = audit_pilot.MetricFeatureExtractor(embed_model)
    # share embeddings cache
    extractor.concept_embeddings = concepts_cache

    cutoff_data = {}
    all_concepts = set()

    for T in [2016, 2018, 2020]:
        G_T, test_co, test_first_paper, test_concepts, p_by_yr_c, _, _ = audit_pilot.build_temporal_split(
            papers, cutoff_T=T, eval_span=1
        )
        candidate_pos = []
        for (u, v), cnt in test_co.items():
            if cnt >= 2 and G_T.has_node(u) and G_T.has_node(v) and not G_T.has_edge(u, v):
                du, dv = G_T.degree(u), G_T.degree(v)
                p1, p2 = (u, v) if du <= dv else (v, u)
                candidate_pos.append((p1, p2))
        random.Random(seed + T).shuffle(candidate_pos)
        positives = candidate_pos[:500]

        nodes_rest = [n for n in G_T.nodes() if n in test_concepts]
        neg_rest = audit_pilot.sample_caliper_matched_negatives(
            G_T, positives, nodes_rest, test_co, 0.15, seed=seed + T + 100
        )
        ds_rest = audit_pilot.build_evaluation_dataset(positives, neg_rest, test_first_paper, seed=seed + T + 300)
        for it in ds_rest:
            all_concepts.add(it["u"])
            all_concepts.add(it["v"])

        cutoff_data[T] = {
            "G_T": G_T, "ds_rest": ds_rest, "p_by_yr_c": p_by_yr_c, "test_first_paper": test_first_paper
        }

    extractor.precompute_embeddings(all_concepts)
    concepts_cache.update(extractor.concept_embeddings)

    for T in [2016, 2018, 2020]:
        G_T = cutoff_data[T]["G_T"]
        p_by_yr_c = cutoff_data[T]["p_by_yr_c"]
        louv = extractor.compute_louvain_structure(G_T, seed=seed)
        cutoff_data[T]["feat_rest"] = extractor.extract_features(
            cutoff_data[T]["ds_rest"], G_T, louv, T, p_by_yr_c
        )

    # Fit on 2016
    train_feat = cutoff_data[2016]["feat_rest"]
    y_train = np.array([r["label"] for r in train_feat], dtype=int)
    def mat(recs, keys):
        return np.column_stack([[r[k] for r in recs] for k in keys])

    keys_ref = ["rn_aa", "rn_log_du", "rn_log_dv"]
    keys_mom = ["rn_mom_vol_min", "rn_mom_vol_max", "rn_mom_growth_min", "rn_mom_growth_max"]
    keys_ref_mom = keys_ref + keys_mom
    keys_ref_both = keys_ref + ["rn_dist_louv", "rn_dist_emb"]
    keys_ref_mom_both = keys_ref + keys_mom + ["rn_dist_louv", "rn_dist_emb"]

    lr_aa_deg = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed).fit(mat(train_feat, keys_ref), y_train)
    lr_deg_only = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed).fit(mat(train_feat, ["rn_log_du", "rn_log_dv"]), y_train)
    lr_ref_mom = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed).fit(mat(train_feat, keys_ref_mom), y_train)
    lr_ref_both = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed).fit(mat(train_feat, keys_ref_both), y_train)
    lr_ref_mom_both = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed).fit(mat(train_feat, keys_ref_mom_both), y_train)

    tau_aa = np.percentile([r["rn_aa"] for r in train_feat], 75.0)
    tau_emb = np.percentile([r["rn_dist_emb"] for r in train_feat], 25.0)

    d_auc_results = {}
    pooled_first_papers = []

    for T in [2016, 2018, 2020]:
        feat = cutoff_data[T]["feat_rest"]
        y_eval = np.array([r["label"] for r in feat], dtype=int)

        p_deg_only      = lr_deg_only.predict_proba(mat(feat, ["rn_log_du", "rn_log_dv"]))[:, 1]
        p_aa_deg        = lr_aa_deg.predict_proba(mat(feat, keys_ref))[:, 1]
        p_ref_mom       = lr_ref_mom.predict_proba(mat(feat, keys_ref_mom))[:, 1]
        p_ref_both      = lr_ref_both.predict_proba(mat(feat, keys_ref_both))[:, 1]
        p_ref_mom_both  = lr_ref_mom_both.predict_proba(mat(feat, keys_ref_mom_both))[:, 1]

        scores_map = {
            "Degree-only Logistic":                     p_deg_only,
            "AA Alone":                                 np.array([r["rn_aa"] for r in feat]),
            "Jaccard":                                  np.array([r["rn_jaccard"] for r in feat]),
            "Resource Allocation":                      np.array([r["rn_ra"] for r in feat]),
            "Salton Cosine":                            np.array([r["rn_salton"] for r in feat]),
            "AA + Degree (Reference Baseline)":         p_aa_deg,
            "Louvain Dist Only":                        np.array([-r["rn_dist_louv"] for r in feat]),
            "Embedding Dist Only":                      np.array([-r["rn_dist_emb"] for r in feat]),
            "Reference Baseline + Momentum":            p_ref_mom,
            "Reference + Both Distances":               p_ref_both,
            "Reference + Momentum + Both Distances":    p_ref_mom_both,
            "Hard Gate: AA Plausibility (Ablation)":    np.array([(1.0 - r["rn_dist_emb"]) if r["rn_aa"] >= tau_aa else 0.0 for r in feat]),
            "Hard Gate: Same Community (Ablation)":     np.array([r["rn_aa"] if r["dist_louv"] == 0.0 else 0.0 for r in feat]),
            "Hard Gate: Embedding Dist <= tau (Abl.)":  np.array([r["rn_aa"] if r["rn_dist_emb"] <= tau_emb else 0.0 for r in feat]),
        }

        ref_auc = fast_auroc(y_eval, p_aa_deg)
        d_auc_map = {}
        for m, sc in scores_map.items():
            d_auc_map[m] = fast_auroc(y_eval, sc) - ref_auc
        d_auc_results[T] = d_auc_map

        if seed == 42:
            paper_grp = defaultdict(list)
            for idx, r in enumerate(feat):
                r["score_ref"]          = p_aa_deg[idx]
                r["score_ref_both"]     = p_ref_both[idx]
                r["score_ref_mom"]      = p_ref_mom[idx]
                r["score_ref_mom_both"] = p_ref_mom_both[idx]
                if r["label"] == 1 and r["paper_id"] != "NA":
                    paper_grp[r["paper_id"]].append(r)
            for pid, recs in paper_grp.items():
                first_p = min(recs, key=lambda x: (x["year"], x["paper_id"]))
                pooled_first_papers.append({
                    "paper_id": pid,
                    "year": first_p["year"],
                    "citations": first_p["citations"],
                    "pct_rank": first_p["pct_rank"],
                    "score_ref": max(r["score_ref"] for r in recs),
                    "score_ref_both": max(r["score_ref_both"] for r in recs),
                    "score_ref_mom": max(r["score_ref_mom"] for r in recs),
                    "score_ref_mom_both": max(r["score_ref_mom_both"] for r in recs),
                    "rn_aa": max(r["rn_aa"] for r in recs),
                    "rn_dist_louv": max(r["rn_dist_louv"] for r in recs),
                    "rn_dist_emb": max(r["rn_dist_emb"] for r in recs),
                })

    return d_auc_results, pooled_first_papers

print("Computing Table 5 across seeds 42, 43, 44...", flush=True)
d42, papers42 = get_seed_d_aucs(42)
d43, _ = get_seed_d_aucs(43)
d44, _ = get_seed_d_aucs(44)

print("\n" + "=" * 136, flush=True)
print("TABLE 5: MULTI-SEED REPLICATION (SEEDS 42, 43, 44): ΔAUROC VS. REFERENCE BASELINE AT 2018 AND 2020", flush=True)
print("         (MODELS FITTED ON T=2016 PER SEED, FROZEN, AND EVALUATED ON T=2018 AND T=2020)", flush=True)
print("=" * 136, flush=True)
hdr_t5 = f"{'Cutoff':<8} │ {'Model':<42} │ {'ΔAUROC (42)':>14} │ {'ΔAUROC (43)':>14} │ {'ΔAUROC (44)':>14} │ {'Mean ± SD':^20}"
sep_t5 = "─" * 9 + "┼" + "─" * 44 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 22
print(hdr_t5, flush=True)
print(sep_t5, flush=True)

models_list = list(d42[2018].keys())
for T in [2018, 2020]:
    for m in models_list:
        v42 = d42[T][m]
        v43 = d43[T][m]
        v44 = d44[T][m]
        arr = np.array([v42, v43, v44])
        mean_sd = f"{np.mean(arr):+.4f} ± {np.std(arr, ddof=1):.4f}"
        print(f"T={T:<6} │ {m:<42} │ {v42:>+14.4f} │ {v43:>+14.4f} │ {v44:>+14.4f} │ {mean_sd:^20}", flush=True)
    print(sep_t5, flush=True)

print("\n" + "=" * 132, flush=True)
print("TABLE 6: RQ2 WITHIN-YEAR CITATION PERCENTILE RANK REGRESSED ON NOVELTY PROBABILITIES", flush=True)
print("         (AGGREGATED TO ONE OBSERVATION PER FIRST-REALIZING PAPER, YEAR FIXED EFFECTS, CLUSTERED SE)", flush=True)
print("=" * 132, flush=True)
print("PER-PAPER AGGREGATION RULE:", flush=True)
print("  - Each unique paper p that introduces one or more realized novel pairs constitutes exactly one row.", flush=True)
print("  - Paper novelty score = max predicted probability among novel pairs realized in p.", flush=True)
print("  - Outcome = within-year citation percentile rank of paper p.", flush=True)
print("  - Primary specification: Reference Baseline Probability (Standardized, unadjusted 95% CI).", flush=True)
print("  - Exploratory specifications: Bonferroni-adjusted family-wise confidence intervals and p-values.", flush=True)
print("=" * 132, flush=True)
hdr_t6 = f"{'Predictor Variable':<52} │ {'Role':<12} │ {'Beta':>10} │ {'95% CI (Adj.)':^24} │ {'Clust SE':>9} │ {'t-stat':>8} │ {'Adj. p':>9} │ {'N':>6}"
sep_t6 = "─" * 53 + "┼" + "─" * 14 + "┼" + "─" * 12 + "┼" + "─" * 26 + "┼" + "─" * 11 + "┼" + "─" * 10 + "┼" + "─" * 11 + "┼" + "─" * 8
print(hdr_t6, flush=True)
print(sep_t6, flush=True)

rq2_table = audit_pilot.run_rq2_paper_regressions(papers42, "score_ref", "Reference Baseline Probability")
for r in rq2_table:
    b_str = f"{r['beta']:+.4f}"
    ci_str = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
    se_str = f"{r['se']:.4f}"
    t_str = f"{r['t']:+.2f}"
    p_str = f"{r['adj_p']:.4f}" if r["adj_p"] >= 0.0001 else "<0.0001"
    print(f"{r['predictor']:<52} │ {r['role']:<12} │ {b_str:>10} │ {ci_str:^24} │ {se_str:>9} │ {t_str:>8} │ {p_str:>9} │ {r['n']:>6,d}", flush=True)
print("=" * 132, flush=True)
