"""
run_final_pipeline.py — End-to-end execution of all 7 audit tasks and CSV export.
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import math, json, random, time, warnings, csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
import numpy as np
import scipy.stats as stats
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# METADATA & CONFIG
# ──────────────────────────────────────────────────────────────────────────────
GIT_COMMIT = "e9b1c053dce1ab6cf99129d3f8ba2ec8545ee637"
DATA_SNAPSHOT_DATE = "2026-10-11"

CFG = dict(
    seeds         = [42, 43, 44],
    primary_seed  = 42,
    train_cutoff  = 2016,
    eval_cutoffs  = [2016, 2018, 2020],
    eval_span     = 1,
    n_pos         = 500,
    n_neg         = 500,
    n_boot        = 1000,
    caliper_pct   = 0.15,
    gate_pct      = 0.25,
)

# Fast AUROC calculation
def fast_auroc(y_true, scores):
    y_arr = np.asarray(y_true)
    pos = (y_arr == 1)
    n_pos = int(np.sum(pos))
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(scores)
    u = np.sum(ranks[pos]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))

def fast_ap(y_true, scores):
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# DATASET & GRAPH UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def build_temporal_split(papers, cutoff_T, eval_span=1):
    max_test_year = cutoff_T + eval_span
    train_papers = [p for p in papers if p["year"] < cutoff_T]
    test_papers  = [p for p in papers if cutoff_T <= p["year"] <= max_test_year]

    papers_by_year_concept = defaultdict(lambda: defaultdict(int))
    for p in train_papers:
        yr = p["year"]
        for c in set(p["concepts"]):
            papers_by_year_concept[yr][c] += 1

    G_T = nx.Graph()
    for p in train_papers:
        cs = set(p["concepts"])
        for c in cs:
            if not G_T.has_node(c):
                G_T.add_node(c, count=0)
            G_T.nodes[c]["count"] += 1
        for u, v in combinations(sorted(cs), 2):
            if G_T.has_edge(u, v):
                G_T[u][v]["weight"] += 1
            else:
                G_T.add_edge(u, v, weight=1)

    test_co = defaultdict(int)
    test_first_paper = {}
    test_concepts = set()

    for p in sorted(test_papers, key=lambda x: (x["year"], x["id"])):
        cs = sorted(set(p["concepts"]))
        pyr = p["year"]
        cit = p.get("citations", 0)
        pct = p.get("pct_rank", 0.0)
        pid = p["id"]

        for c in cs:
            if G_T.has_node(c):
                test_concepts.add(c)

        for u, v in combinations(cs, 2):
            pair_k = (u, v)
            test_co[pair_k] += 1
            if pair_k not in test_first_paper:
                test_first_paper[pair_k] = {
                    "paper_id": pid, "year": pyr, "citations": cit, "pct_rank": pct
                }

    return G_T, dict(test_co), test_first_paper, test_concepts, papers_by_year_concept, len(train_papers), len(test_papers)


def sample_caliper_matched_negatives(G_T, positives, candidate_nodes, test_co, caliper_pct, seed):
    rng = random.Random(seed)
    deg_map = defaultdict(list)
    for n in candidate_nodes:
        deg_map[G_T.degree(n)].append(n)

    all_degs = sorted(deg_map.keys())
    negatives = []
    used = set()

    for pu, pv in positives:
        t_du = G_T.degree(pu)
        t_dv = G_T.degree(pv)
        tol_u = max(1, int(round(caliper_pct * t_du)))
        tol_v = max(1, int(round(caliper_pct * t_dv)))

        pool_u = [n for d in all_degs if abs(d - t_du) <= tol_u for n in deg_map[d]]
        pool_v = [n for d in all_degs if abs(d - t_dv) <= tol_v for n in deg_map[d]]

        if not pool_u or not pool_v:
            continue

        matched = False
        for _ in range(50):
            nu = rng.choice(pool_u)
            nv = rng.choice(pool_v)
            if nu == nv:
                continue

            pair_k = (min(nu, nv), max(nu, nv))
            if pair_k in used:
                continue
            if G_T.has_edge(nu, nv):
                continue
            if test_co.get(pair_k, 0) > 0:
                continue

            d_nu, d_nv = G_T.degree(nu), G_T.degree(nv)
            p1, p2 = (nu, nv) if d_nu <= d_nv else (nv, nu)
            negatives.append((p1, p2))
            used.add(pair_k)
            matched = True
            break

    return negatives


def build_evaluation_dataset(positives, negatives, test_first_paper, seed):
    rng = random.Random(seed)
    pos_recs = []
    for u, v in positives:
        pk = (min(u, v), max(u, v))
        pinfo = test_first_paper.get(pk, {"paper_id": "NA", "year": 0, "citations": 0, "pct_rank": 0.0})
        pos_recs.append({
            "u": u, "v": v, "label": 1,
            "paper_id": pinfo["paper_id"],
            "year": pinfo["year"],
            "citations": pinfo["citations"],
            "pct_rank": pinfo["pct_rank"],
        })
    neg_recs = [
        {"u": u, "v": v, "label": 0, "paper_id": "NEG", "year": 0, "citations": 0, "pct_rank": 0.0}
        for u, v in negatives
    ]
    ds = pos_recs + neg_recs
    rng.shuffle(ds)
    return ds

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────
class GlobalFeatureExtractor:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.concept_embeddings = {}

    def ensure_embeddings(self, concepts_list):
        missing = [c for c in concepts_list if c not in self.concept_embeddings]
        if missing:
            with torch.no_grad():
                embs = self.embed_model.encode(
                    missing, batch_size=32, show_progress_bar=False, normalize_embeddings=True
                )
            for c, vec in zip(missing, embs):
                self.concept_embeddings[c] = vec

    def compute_louvain_structure(self, G_T, seed=42):
        communities = nx.community.louvain_communities(G_T, seed=seed, weight="weight")
        node_to_comm = {}
        comm_sizes = {}
        for c_idx, comm_nodes in enumerate(communities):
            comm_sizes[c_idx] = len(comm_nodes)
            for n in comm_nodes:
                node_to_comm[n] = c_idx

        inter_edges = defaultdict(int)
        for u, v in G_T.edges():
            cu = node_to_comm.get(u)
            cv = node_to_comm.get(v)
            if cu is not None and cv is not None and cu != cv:
                inter_edges[(min(cu, cv), max(cu, cv))] += 1

        inter_density = {}
        max_density = 0.0
        for (c1, c2), count in inter_edges.items():
            possible = comm_sizes[c1] * comm_sizes[c2]
            dens = count / possible if possible > 0 else 0.0
            inter_density[(c1, c2)] = dens
            if dens > max_density:
                max_density = dens

        return {
            "node_to_comm": node_to_comm,
            "inter_density": inter_density,
            "max_density": max_density if max_density > 0 else 1.0,
        }

    def compute_concept_momentum(self, G_T, cutoff_T, papers_by_year_concept):
        n_prev1 = papers_by_year_concept[cutoff_T - 1]
        n_prev2 = papers_by_year_concept[cutoff_T - 2]

        all_nodes = list(G_T.nodes())
        vols = [n_prev1.get(c, 0) for c in all_nodes]
        growths = [(n_prev1.get(c, 0) + 1.0) / (n_prev2.get(c, 0) + 1.0) for c in all_nodes]

        ranks_vol = stats.rankdata(vols, method="average") / len(all_nodes)
        ranks_growth = stats.rankdata(growths, method="average") / len(all_nodes)

        concept_rn_vol = {c: float(r) for c, r in zip(all_nodes, ranks_vol)}
        concept_rn_growth = {c: float(r) for c, r in zip(all_nodes, ranks_growth)}
        return concept_rn_vol, concept_rn_growth

    def extract_features(self, dataset, G_T, louv_struct, cutoff_T, papers_by_year_concept):
        node_to_comm = louv_struct["node_to_comm"]
        inter_density = louv_struct["inter_density"]
        max_density = louv_struct["max_density"]

        c_vol, c_growth = self.compute_concept_momentum(G_T, cutoff_T, papers_by_year_concept)

        scored = []
        for item in dataset:
            u, v = item["u"], item["v"]
            du, dv = G_T.degree(u), G_T.degree(v)

            common = list(nx.common_neighbors(G_T, u, v))
            cn = len(common)

            aa = sum(1.0 / math.log(G_T.degree(w)) for w in common if G_T.degree(w) > 1)
            union_size = du + dv - cn
            jaccard = cn / union_size if union_size > 0 else 0.0
            ra = sum(1.0 / G_T.degree(w) for w in common if G_T.degree(w) > 0)
            salton = cn / math.sqrt(du * dv) if (du > 0 and dv > 0) else 0.0

            log_du = math.log(1.0 + du)
            log_dv = math.log(1.0 + dv)

            cu = node_to_comm.get(u, -1)
            cv = node_to_comm.get(v, -2)
            if cu == cv and cu != -1:
                dist_louv = 0.0
            else:
                dens = inter_density.get((min(cu, cv), max(cu, cv)), 0.0)
                dist_louv = 1.0 + (dens / max_density)

            vec_u = self.concept_embeddings[u]
            vec_v = self.concept_embeddings[v]
            cos_sim = float(np.dot(vec_u, vec_v))
            dist_emb = max(0.0, 1.0 - cos_sim)

            vu, vv = c_vol.get(u, 0.5), c_vol.get(v, 0.5)
            gu, gv = c_growth.get(u, 0.5), c_growth.get(v, 0.5)
            mom_vol_min = min(vu, vv)
            mom_vol_max = max(vu, vv)
            mom_growth_min = min(gu, gv)
            mom_growth_max = max(gu, gv)

            scored.append({
                **item,
                "aa": aa,
                "jaccard": jaccard,
                "ra": ra,
                "salton": salton,
                "log_du": log_du,
                "log_dv": log_dv,
                "dist_louv": dist_louv,
                "dist_emb": dist_emb,
                "mom_vol_min": mom_vol_min,
                "mom_vol_max": mom_vol_max,
                "mom_growth_min": mom_growth_min,
                "mom_growth_max": mom_growth_max,
            })

        n = len(scored)
        raw_keys = [
            "aa", "jaccard", "ra", "salton", "log_du", "log_dv",
            "dist_louv", "dist_emb",
            "mom_vol_min", "mom_vol_max", "mom_growth_min", "mom_growth_max"
        ]
        for k in raw_keys:
            vals = [r[k] for r in scored]
            ranks = stats.rankdata(vals, method="average") / n
            for idx, r in enumerate(scored):
                r[f"rn_{k}"] = float(ranks[idx])

        return scored

# ──────────────────────────────────────────────────────────────────────────────
# STATISTICAL INFERENCE & BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
def fit_logistic_with_inference(X, y, feature_names, penalty_C=1.0, seed=42):
    clf = LogisticRegression(penalty="l2", C=penalty_C, solver="lbfgs", max_iter=1000, random_state=seed).fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    w = p * (1.0 - p)

    X_aug = np.column_stack([np.ones(len(X)), X])
    ridge = (1.0 / penalty_C) * np.eye(X_aug.shape[1])
    ridge[0, 0] = 0.0
    H = (X_aug.T * w) @ X_aug + ridge
    cov = np.linalg.inv(H)
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))

    coefs = np.concatenate([[clf.intercept_[0]], clf.coef_[0]])
    names = ["Intercept"] + feature_names

    summary = []
    for nm, b, s in zip(names, coefs, se):
        ci = (b - 1.96 * s, b + 1.96 * s)
        z = b / s if s > 0 else 0.0
        pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        summary.append({
            "feature": nm, "coef": b, "se": s, "ci": ci, "z": z, "p": pval,
        })
    return clf, summary

def evaluate_models_with_bootstrap(y_true, model_scores, ref_baseline_key, n_boot=1000, seed=42):
    n = len(y_true)
    y_true = np.asarray(y_true, dtype=int)
    rng = np.random.RandomState(seed)

    ref_scores = model_scores[ref_baseline_key]
    ref_auc_pt = fast_auroc(y_true, ref_scores)
    y_shuffled = rng.permutation(y_true)

    results = {}
    for m_name, scores in model_scores.items():
        results[m_name] = {
            "auc": fast_auroc(y_true, scores),
            "ap": fast_ap(y_true, scores),
            "shuf_auc": fast_auroc(y_shuffled, scores),
            "d_auc": fast_auroc(y_true, scores) - ref_auc_pt,
            "auc_boots": [], "ap_boots": [], "shuf_auc_boots": [], "d_auc_boots": [],
        }

    boot_indices = rng.randint(0, n, size=(n_boot, n))
    for b in range(n_boot):
        b_idx = boot_indices[b]
        b_y = y_true[b_idx]
        b_yshuf = y_shuffled[b_idx]
        if np.sum(b_y) == 0 or np.sum(b_y) == n:
            continue
        b_ref_auc = fast_auroc(b_y, ref_scores[b_idx])

        for m_name, scores in model_scores.items():
            b_sc = scores[b_idx]
            b_auc = fast_auroc(b_y, b_sc)
            results[m_name]["auc_boots"].append(b_auc)
            results[m_name]["ap_boots"].append(fast_ap(b_y, b_sc))
            results[m_name]["shuf_auc_boots"].append(fast_auroc(b_yshuf, b_sc))
            results[m_name]["d_auc_boots"].append(b_auc - b_ref_auc)

    final_table = []
    for m_name, d in results.items():
        auc_lo, auc_hi = np.percentile(d["auc_boots"], [2.5, 97.5])
        ap_lo,  ap_hi  = np.percentile(d["ap_boots"],  [2.5, 97.5])
        sh_lo,  sh_hi  = np.percentile(d["shuf_auc_boots"], [2.5, 97.5])
        d_lo,   d_hi   = np.percentile(d["d_auc_boots"], [2.5, 97.5])

        final_table.append({
            "model": m_name,
            "auc": d["auc"], "auc_ci": (auc_lo, auc_hi),
            "ap": d["ap"],   "ap_ci": (ap_lo, ap_hi),
            "shuf_auc": d["shuf_auc"], "shuf_auc_ci": (sh_lo, sh_hi),
            "d_auc": d["d_auc"], "d_auc_ci": (d_lo, d_hi),
        })
    return final_table

# ──────────────────────────────────────────────────────────────────────────────
# RQ2 REGRESSION UTILITY
# ──────────────────────────────────────────────────────────────────────────────
def run_rq2_paper_regressions(first_realizing_papers_list, primary_key="score_ref", primary_name="Reference Baseline Probability"):
    n = len(first_realizing_papers_list)
    if n < 10:
        return []

    years = np.array([p["year"] for p in first_realizing_papers_list])
    y_raw = np.array([p["pct_rank"] for p in first_realizing_papers_list])

    unique_years = np.unique(years)
    k_years = len(unique_years)
    y_demeaned = np.zeros(n)
    for y in unique_years:
        mask = (years == y)
        y_demeaned[mask] = y_raw[mask] - np.mean(y_raw[mask])

    predictors = [
        (f"{primary_name} (Standardized)", primary_key, "Primary"),
        ("Reference + Both Distances (Standardized)", "score_ref_both", "Exploratory"),
        ("Reference + Momentum (Standardized)", "score_ref_mom", "Exploratory"),
        ("Reference + Mom + Both Distances (Standardized)", "score_ref_mom_both", "Exploratory"),
        ("Adamic-Adar Alone (Standardized)", "rn_aa", "Exploratory"),
        ("Louvain Community Distance (Standardized)", "rn_dist_louv", "Exploratory"),
        ("Concept Embedding Distance (Standardized)", "rn_dist_emb", "Exploratory"),
    ]

    m_exploratory = sum(1 for _, _, role in predictors if role == "Exploratory")
    z_bonf = float(stats.norm.ppf(1.0 - (0.05 / (2.0 * max(1, m_exploratory)))))

    records = []
    for label, key, role in predictors:
        raw_x = np.array([p[key] for p in first_realizing_papers_list])
        std_x = (raw_x - np.mean(raw_x)) / (np.std(raw_x) + 1e-12)

        x_demeaned = np.zeros(n)
        for y in unique_years:
            mask = (years == y)
            x_demeaned[mask] = std_x[mask] - np.mean(std_x[mask])

        ss_x = np.sum(x_demeaned ** 2)
        if ss_x < 1e-12:
            continue

        beta = np.sum(x_demeaned * y_demeaned) / ss_x
        residuals = y_demeaned - beta * x_demeaned

        df_adj = (n - 1) / max(1, n - k_years)
        meat = np.sum((x_demeaned * residuals) ** 2)
        v_clust = df_adj * meat / (ss_x ** 2)
        se_clust = math.sqrt(max(0.0, v_clust))

        z_multiplier = 1.96 if role == "Primary" else z_bonf
        ci = (beta - z_multiplier * se_clust, beta + z_multiplier * se_clust)
        t_stat = beta / se_clust if se_clust > 0 else 0.0

        raw_pval = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, n - k_years)))
        adj_pval = raw_pval if role == "Primary" else min(1.0, m_exploratory * raw_pval)

        records.append({
            "predictor": label,
            "role": role,
            "beta": beta,
            "se": se_clust,
            "ci": ci,
            "t": t_stat,
            "raw_p": raw_pval,
            "adj_p": adj_pval,
            "n": n,
        })
    return records

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER FOR ONE SEED
# ──────────────────────────────────────────────────────────────────────────────
def run_seed_pipeline(seed, extractor):
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

    cutoff_data = {}
    all_concepts = set()

    for T in CFG["eval_cutoffs"]:
        G_T, test_co, test_first_paper, test_concepts, p_by_yr_c, n_tr, n_te = build_temporal_split(
            papers, cutoff_T=T, eval_span=CFG["eval_span"]
        )

        candidate_pos = []
        for (u, v), cnt in test_co.items():
            if cnt >= 2 and G_T.has_node(u) and G_T.has_node(v) and not G_T.has_edge(u, v):
                du, dv = G_T.degree(u), G_T.degree(v)
                p1, p2 = (u, v) if du <= dv else (v, u)
                candidate_pos.append((p1, p2))
        random.Random(seed + T).shuffle(candidate_pos)
        positives = candidate_pos[:CFG["n_pos"]]

        nodes_rest = [n for n in G_T.nodes() if n in test_concepts]
        nodes_unrest = list(G_T.nodes())

        neg_rest = sample_caliper_matched_negatives(
            G_T, positives, nodes_rest, test_co, CFG["caliper_pct"], seed=seed + T + 100
        )
        neg_unrest = sample_caliper_matched_negatives(
            G_T, positives, nodes_unrest, test_co, CFG["caliper_pct"], seed=seed + T + 200
        )

        ds_rest = build_evaluation_dataset(positives, neg_rest, test_first_paper, seed=seed + T + 300)
        ds_unrest = build_evaluation_dataset(positives, neg_unrest, test_first_paper, seed=seed + T + 400)

        for it in ds_rest + ds_unrest:
            all_concepts.add(it["u"])
            all_concepts.add(it["v"])

        cutoff_data[T] = {
            "G_T": G_T,
            "test_co": test_co,
            "positives": positives,
            "neg_rest": neg_rest,
            "neg_unrest": neg_unrest,
            "ds_rest": ds_rest,
            "ds_unrest": ds_unrest,
            "nodes_rest_count": len(nodes_rest),
            "nodes_unrest_count": len(nodes_unrest),
            "p_by_yr_c": p_by_yr_c,
        }

    extractor.ensure_embeddings(list(all_concepts))

    for T in CFG["eval_cutoffs"]:
        G_T = cutoff_data[T]["G_T"]
        p_by_yr_c = cutoff_data[T]["p_by_yr_c"]
        louv = extractor.compute_louvain_structure(G_T, seed=seed)
        cutoff_data[T]["feat_rest"] = extractor.extract_features(
            cutoff_data[T]["ds_rest"], G_T, louv, T, p_by_yr_c
        )
        cutoff_data[T]["feat_unrest"] = extractor.extract_features(
            cutoff_data[T]["ds_unrest"], G_T, louv, T, p_by_yr_c
        )

    # Models fitted on T=2016
    train_feat = cutoff_data[2016]["feat_rest"]
    y_train = np.array([r["label"] for r in train_feat], dtype=int)
    def mat(recs, keys):
        return np.column_stack([[r[k] for r in recs] for k in keys])

    keys_ref = ["rn_aa", "rn_log_du", "rn_log_dv"]
    keys_mom = ["rn_mom_vol_min", "rn_mom_vol_max", "rn_mom_growth_min", "rn_mom_growth_max"]
    keys_ref_mom = keys_ref + keys_mom
    keys_ref_both = keys_ref + ["rn_dist_louv", "rn_dist_emb"]
    keys_ref_mom_both = keys_ref + keys_mom + ["rn_dist_louv", "rn_dist_emb"]

    lr_aa_deg, sum_aa_deg = fit_logistic_with_inference(mat(train_feat, keys_ref), y_train, keys_ref, seed=seed)
    lr_deg_only, sum_deg_only = fit_logistic_with_inference(mat(train_feat, ["rn_log_du", "rn_log_dv"]), y_train, ["rn_log_du", "rn_log_dv"], seed=seed)
    lr_ref_mom, sum_ref_mom = fit_logistic_with_inference(mat(train_feat, keys_ref_mom), y_train, keys_ref_mom, seed=seed)
    lr_ref_both, sum_ref_both = fit_logistic_with_inference(mat(train_feat, keys_ref_both), y_train, keys_ref_both, seed=seed)
    lr_ref_mom_both, sum_ref_mom_both = fit_logistic_with_inference(mat(train_feat, keys_ref_mom_both), y_train, keys_ref_mom_both, seed=seed)

    tau_aa = np.percentile([r["rn_aa"] for r in train_feat], 75.0)
    tau_emb = np.percentile([r["rn_dist_emb"] for r in train_feat], 25.0)

    # Evaluate on both Restricted and Unrestricted
    eval_rest = {}
    eval_unrest = {}
    pooled_first_papers = []

    for T in CFG["eval_cutoffs"]:
        for mode, target_eval_dict in [("rest", eval_rest), ("unrest", eval_unrest)]:
            feat = cutoff_data[T][f"feat_{mode}"]
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

            if mode == "rest":
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

            summary = evaluate_models_with_bootstrap(
                y_eval, scores_map, ref_baseline_key="AA + Degree (Reference Baseline)",
                n_boot=CFG["n_boot"], seed=seed + T
            )
            target_eval_dict[T] = summary

    log_coef_tables = {
        "Degree-only Logistic": sum_deg_only,
        "AA + Degree (Reference Baseline)": sum_aa_deg,
        "Reference Baseline + Momentum": sum_ref_mom,
        "Reference + Both Distances": sum_ref_both,
        "Reference + Momentum + Both Distances": sum_ref_mom_both,
    }

    return {
        "papers": papers,
        "cutoff_data": cutoff_data,
        "eval_rest": eval_rest,
        "eval_unrest": eval_unrest,
        "log_coef_tables": log_coef_tables,
        "pooled_first_papers": pooled_first_papers,
        "models_fitted": {
            "lr_aa_deg": lr_aa_deg,
            "lr_deg_only": lr_deg_only,
            "lr_ref_mom": lr_ref_mom,
            "lr_ref_both": lr_ref_both,
            "lr_ref_mom_both": lr_ref_mom_both,
            "keys_ref": keys_ref,
            "keys_ref_both": keys_ref_both,
        }
    }

# ──────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS (REQUIREMENT 2)
# ──────────────────────────────────────────────────────────────────────────────
def run_sensitivity_analysis(seed42_res):
    """
    Fit Ref and Ref+Both Distances on T=2018 data, evaluate on T=2020.
    Paired bootstrap CI of ΔAUROC and ΔAP.
    """
    cdata = seed42_res["cutoff_data"]
    feat_2018 = cdata[2018]["feat_rest"]
    y_2018 = np.array([r["label"] for r in feat_2018], dtype=int)

    def mat(recs, keys):
        return np.column_stack([[r[k] for r in recs] for k in keys])

    keys_ref = ["rn_aa", "rn_log_du", "rn_log_dv"]
    keys_ref_both = keys_ref + ["rn_dist_louv", "rn_dist_emb"]

    lr_ref_2018 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42).fit(
        mat(feat_2018, keys_ref), y_2018
    )
    lr_both_2018 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42).fit(
        mat(feat_2018, keys_ref_both), y_2018
    )

    feat_2020 = cdata[2020]["feat_rest"]
    y_2020 = np.array([r["label"] for r in feat_2020], dtype=int)
    p_ref_2020 = lr_ref_2018.predict_proba(mat(feat_2020, keys_ref))[:, 1]
    p_both_2020 = lr_both_2018.predict_proba(mat(feat_2020, keys_ref_both))[:, 1]

    auc_ref = fast_auroc(y_2020, p_ref_2020)
    auc_both = fast_auroc(y_2020, p_both_2020)
    d_auc = auc_both - auc_ref

    ap_ref = fast_ap(y_2020, p_ref_2020)
    ap_both = fast_ap(y_2020, p_both_2020)
    d_ap = ap_both - ap_ref

    n = len(y_2020)
    rng = np.random.RandomState(42)
    boot_indices = rng.randint(0, n, size=(1000, n))

    d_auc_boots = []
    d_ap_boots = []
    auc_ref_boots = []
    auc_both_boots = []
    ap_ref_boots = []
    ap_both_boots = []

    for b in range(1000):
        b_idx = boot_indices[b]
        b_y = y_2020[b_idx]
        if np.sum(b_y) == 0 or np.sum(b_y) == n:
            continue
        b_p_ref = p_ref_2020[b_idx]
        b_p_both = p_both_2020[b_idx]

        b_auc_r = fast_auroc(b_y, b_p_ref)
        b_auc_b = fast_auroc(b_y, b_p_both)
        b_ap_r = fast_ap(b_y, b_p_ref)
        b_ap_b = fast_ap(b_y, b_p_both)

        auc_ref_boots.append(b_auc_r)
        auc_both_boots.append(b_auc_b)
        ap_ref_boots.append(b_ap_r)
        ap_both_boots.append(b_ap_b)
        d_auc_boots.append(b_auc_b - b_auc_r)
        d_ap_boots.append(b_ap_b - b_ap_r)

    auc_ref_ci = tuple(np.percentile(auc_ref_boots, [2.5, 97.5]))
    auc_both_ci = tuple(np.percentile(auc_both_boots, [2.5, 97.5]))
    d_auc_ci = tuple(np.percentile(d_auc_boots, [2.5, 97.5]))

    ap_ref_ci = tuple(np.percentile(ap_ref_boots, [2.5, 97.5]))
    ap_both_ci = tuple(np.percentile(ap_both_boots, [2.5, 97.5]))
    d_ap_ci = tuple(np.percentile(d_ap_boots, [2.5, 97.5]))

    return {
        "train_cutoff": 2018,
        "eval_cutoff": 2020,
        "auc_ref": auc_ref, "auc_ref_ci": auc_ref_ci,
        "auc_both": auc_both, "auc_both_ci": auc_both_ci,
        "d_auc": d_auc, "d_auc_ci": d_auc_ci,
        "ap_ref": ap_ref, "ap_ref_ci": ap_ref_ci,
        "ap_both": ap_both, "ap_both_ci": ap_both_ci,
        "d_ap": d_ap, "d_ap_ci": d_ap_ci,
    }

# ──────────────────────────────────────────────────────────────────────────────
# NATURAL-CANDIDATES CHECK (REQUIREMENT 4)
# ──────────────────────────────────────────────────────────────────────────────
def run_natural_candidates_check(seed42_res, extractor, sample_size=5000):
    cdata = seed42_res["cutoff_data"]
    models = seed42_res["models_fitted"]
    lr_aa_deg = models["lr_aa_deg"]
    lr_ref_both = models["lr_ref_both"]
    keys_ref = models["keys_ref"]
    keys_ref_both = models["keys_ref_both"]

    results = []

    for T in CFG["eval_cutoffs"]:
        G_T = cdata[T]["G_T"]
        test_co = cdata[T]["test_co"]
        all_nodes = list(G_T.nodes())
        n_nodes = len(all_nodes)
        rng = random.Random(42 + T)

        # Sample 5,000 distinct unseen pairs (u, v) uniformly at random
        sampled_pairs = set()
        while len(sampled_pairs) < sample_size:
            i = rng.randint(0, n_nodes - 1)
            j = rng.randint(0, n_nodes - 1)
            if i == j:
                continue
            u, v = (all_nodes[i], all_nodes[j]) if all_nodes[i] < all_nodes[j] else (all_nodes[j], all_nodes[i])
            if (u, v) in sampled_pairs:
                continue
            if G_T.has_edge(u, v):
                continue
            sampled_pairs.add((u, v))

        pairs_list = sorted(list(sampled_pairs))
        concepts_to_embed = set()
        for u, v in pairs_list:
            concepts_to_embed.add(u)
            concepts_to_embed.add(v)
        extractor.ensure_embeddings(list(concepts_to_embed))

        louv = extractor.compute_louvain_structure(G_T, seed=42)
        node_to_comm = louv["node_to_comm"]
        inter_density = louv["inter_density"]
        max_density = louv["max_density"]

        y_true = np.array([1 if test_co.get((u, v), 0) >= 1 else 0 for u, v in pairs_list], dtype=int)
        n_pos = int(np.sum(y_true))
        base_rate = (n_pos / sample_size) * 100.0

        pa_scores = []
        raw_recs = []
        for u, v in pairs_list:
            du, dv = G_T.degree(u), G_T.degree(v)
            pa_scores.append(du * dv)

            common = list(nx.common_neighbors(G_T, u, v))
            aa = sum(1.0 / math.log(G_T.degree(w)) for w in common if G_T.degree(w) > 1)
            log_du = math.log(1.0 + du)
            log_dv = math.log(1.0 + dv)

            cu = node_to_comm.get(u, -1)
            cv = node_to_comm.get(v, -2)
            if cu == cv and cu != -1:
                dist_louv = 0.0
            else:
                dens = inter_density.get((min(cu, cv), max(cu, cv)), 0.0)
                dist_louv = 1.0 + (dens / max_density)

            vec_u = extractor.concept_embeddings[u]
            vec_v = extractor.concept_embeddings[v]
            dist_emb = max(0.0, 1.0 - float(np.dot(vec_u, vec_v)))

            raw_recs.append({
                "aa": aa, "log_du": log_du, "log_dv": log_dv,
                "dist_louv": dist_louv, "dist_emb": dist_emb
            })

        n_p = len(raw_recs)
        for k in ["aa", "log_du", "log_dv", "dist_louv", "dist_emb"]:
            vals = [r[k] for r in raw_recs]
            rn = stats.rankdata(vals, method="average") / n_p
            for idx, r in enumerate(raw_recs):
                r[f"rn_{k}"] = float(rn[idx])

        def mat(recs, keys):
            return np.column_stack([[r[k] for r in recs] for k in keys])

        p_ref = lr_aa_deg.predict_proba(mat(raw_recs, keys_ref))[:, 1]
        p_both = lr_ref_both.predict_proba(mat(raw_recs, keys_ref_both))[:, 1]
        pa_arr = np.array(pa_scores)

        res_cutoff = {
            "cutoff": T,
            "sample_size": sample_size,
            "n_pos": n_pos,
            "base_rate": base_rate,
            "models": {
                "Preferential Attachment (PA)": {
                    "auc": fast_auroc(y_true, pa_arr),
                    "ap": fast_ap(y_true, pa_arr),
                },
                "Reference Baseline (AA + Degree)": {
                    "auc": fast_auroc(y_true, p_ref),
                    "ap": fast_ap(y_true, p_ref),
                },
                "Ref + Both Distances": {
                    "auc": fast_auroc(y_true, p_both),
                    "ap": fast_ap(y_true, p_both),
                },
            }
        }
        results.append(res_cutoff)

    return results

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    extractor = GlobalFeatureExtractor(embed_model)

    os.makedirs("results", exist_ok=True)

    # 1. RUN PRIMARY SEED 42
    print("Executing Seed 42 Pipeline...", flush=True)
    res_42 = run_seed_pipeline(42, extractor)

    # 2. RUN SEEDS 43 AND 44
    print("Executing Seed 43 Pipeline...", flush=True)
    res_43 = run_seed_pipeline(43, extractor)

    print("Executing Seed 44 Pipeline...", flush=True)
    res_44 = run_seed_pipeline(44, extractor)

    all_seed_results = {42: res_42, 43: res_43, 44: res_44}

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 1: OPENALEX PAGE-ORDER CHECK (CSV & CONSOLE)
    # ──────────────────────────────────────────────────────────────────────────
    papers_42 = res_42["papers"]
    n_total = len(papers_42)
    positions = list(range(n_total))
    citations = [p.get("citations", 0) for p in papers_42]
    overall_r, overall_p_r = stats.pearsonr(positions, citations)
    overall_rho, overall_p_rho = stats.spearmanr(positions, citations)

    by_year = defaultdict(list)
    for idx, p in enumerate(papers_42):
        by_year[p["year"]].append((idx, p.get("citations", 0)))

    t1_rows = []
    t1_rows.append({
        "cohort": "Overall Corpus", "n_papers": n_total,
        "pearson_r": overall_r, "pval_r": overall_p_r,
        "spearman_rho": overall_rho, "pval_rho": overall_p_rho,
        "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
    })
    for y in sorted(by_year.keys()):
        yr_items = by_year[y]
        yr_n = len(yr_items)
        yr_pos = list(range(yr_n))
        yr_cites = [c for _, c in yr_items]
        r_val, p_r = stats.pearsonr(yr_pos, yr_cites)
        rho_val, p_rho = stats.spearmanr(yr_pos, yr_cites)
        t1_rows.append({
            "cohort": str(y), "n_papers": yr_n,
            "pearson_r": r_val, "pval_r": p_r,
            "spearman_rho": rho_val, "pval_rho": p_rho,
            "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
        })

    with open("results/table1_openalex_page_order.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t1_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 2: DEGREE-MATCHED SAMPLE PROFILES & POOL SIZES (CSV & CONSOLE)
    # ──────────────────────────────────────────────────────────────────────────
    t2_rows = []
    for T in CFG["eval_cutoffs"]:
        cdata = res_42["cutoff_data"][T]
        G_T = cdata["G_T"]
        groups = [
            ("Positives (Realized Novel Pairs)", cdata["positives"], len(cdata["positives"])),
            ("Negatives (Restricted to Test Window)", cdata["neg_rest"], cdata["nodes_rest_count"]),
            ("Negatives (Unrestricted Concept Graph)", cdata["neg_unrest"], cdata["nodes_unrest_count"]),
        ]
        for g_name, pairs, p_size in groups:
            u_degs = [G_T.degree(u) for u, v in pairs]
            v_degs = [G_T.degree(v) for u, v in pairs]
            p_degs = [G_T.degree(u) * G_T.degree(v) for u, v in pairs]
            t2_rows.append({
                "cutoff": T, "cohort_variant": g_name,
                "n_pairs": len(pairs), "pool_size": p_size,
                "deg_u_mean": float(np.mean(u_degs)), "deg_u_median": float(np.median(u_degs)),
                "deg_v_mean": float(np.mean(v_degs)), "deg_v_median": float(np.median(v_degs)),
                "deg_prod_mean": float(np.mean(p_degs)), "deg_prod_median": float(np.median(p_degs)),
                "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    with open("results/table2_degree_matched_profiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t2_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t2_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 3: LOGISTIC REGRESSION COEFFICIENTS (FITTED ON T=2016, FROZEN)
    # ──────────────────────────────────────────────────────────────────────────
    t3_rows = []
    for m_name, coef_rows in res_42["log_coef_tables"].items():
        for r in coef_rows:
            t3_rows.append({
                "model": m_name,
                "feature": r["feature"],
                "coef_beta": r["coef"],
                "std_err": r["se"],
                "ci_lo": r["ci"][0],
                "ci_hi": r["ci"][1],
                "z_stat": r["z"],
                "pval": r["p"],
                "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    with open("results/table3_logistic_coefficients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t3_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT 1 / TABLE 5: MULTI-SEED REPLICATION (SEEDS 42, 43, 44)
    # ──────────────────────────────────────────────────────────────────────────
    t5_rows = []
    models_to_track = [
        "Degree-only Logistic",
        "AA Alone",
        "Jaccard",
        "Resource Allocation",
        "Salton Cosine",
        "AA + Degree (Reference Baseline)",
        "Louvain Dist Only",
        "Embedding Dist Only",
        "Reference Baseline + Momentum",
        "Reference + Both Distances",
        "Reference + Momentum + Both Distances",
        "Hard Gate: AA Plausibility (Ablation)",
        "Hard Gate: Same Community (Ablation)",
        "Hard Gate: Embedding Dist <= tau (Abl.)",
    ]

    for T in [2018, 2020]:
        ref_ap_42 = next(r["ap"] for r in res_42["eval_rest"][T] if r["model"] == "AA + Degree (Reference Baseline)")
        ref_ap_43 = next(r["ap"] for r in res_43["eval_rest"][T] if r["model"] == "AA + Degree (Reference Baseline)")
        ref_ap_44 = next(r["ap"] for r in res_44["eval_rest"][T] if r["model"] == "AA + Degree (Reference Baseline)")

        for m_name in models_to_track:
            r42 = next(r for r in res_42["eval_rest"][T] if r["model"] == m_name)
            r43 = next(r for r in res_43["eval_rest"][T] if r["model"] == m_name)
            r44 = next(r for r in res_44["eval_rest"][T] if r["model"] == m_name)

            d_auc_42 = r42["d_auc"]
            d_auc_43 = r43["d_auc"]
            d_auc_44 = r44["d_auc"]
            arr_auc = np.array([d_auc_42, d_auc_43, d_auc_44])

            d_ap_42 = r42["ap"] - ref_ap_42
            d_ap_43 = r43["ap"] - ref_ap_43
            d_ap_44 = r44["ap"] - ref_ap_44
            arr_ap = np.array([d_ap_42, d_ap_43, d_ap_44])

            t5_rows.append({
                "cutoff": T, "model": m_name,
                "d_auroc_seed42": d_auc_42, "d_auroc_seed43": d_auc_43, "d_auroc_seed44": d_auc_44,
                "d_auroc_mean": float(np.mean(arr_auc)), "d_auroc_sd": float(np.std(arr_auc, ddof=1)),
                "d_ap_seed42": d_ap_42, "d_ap_seed43": d_ap_43, "d_ap_seed44": d_ap_44,
                "d_ap_mean": float(np.mean(arr_ap)), "d_ap_sd": float(np.std(arr_ap, ddof=1)),
                "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    with open("results/table5_multiseed_replication.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t5_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t5_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT 2: SENSITIVITY (FIT T=2018, EVAL T=2020)
    # ──────────────────────────────────────────────────────────────────────────
    sens_res = run_sensitivity_analysis(res_42)
    sens_rows = [
        {
            "train_cutoff": sens_res["train_cutoff"],
            "eval_cutoff": sens_res["eval_cutoff"],
            "model": "AA + Degree (Reference Baseline)",
            "auroc": sens_res["auc_ref"],
            "auroc_ci_lo": sens_res["auc_ref_ci"][0],
            "auroc_ci_hi": sens_res["auc_ref_ci"][1],
            "ap": sens_res["ap_ref"],
            "ap_ci_lo": sens_res["ap_ref_ci"][0],
            "ap_ci_hi": sens_res["ap_ref_ci"][1],
            "delta_auroc_vs_ref": 0.0,
            "delta_auroc_ci_lo": 0.0,
            "delta_auroc_ci_hi": 0.0,
            "delta_ap_vs_ref": 0.0,
            "delta_ap_ci_lo": 0.0,
            "delta_ap_ci_hi": 0.0,
            "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
        },
        {
            "train_cutoff": sens_res["train_cutoff"],
            "eval_cutoff": sens_res["eval_cutoff"],
            "model": "Reference + Both Distances",
            "auroc": sens_res["auc_both"],
            "auroc_ci_lo": sens_res["auc_both_ci"][0],
            "auroc_ci_hi": sens_res["auc_both_ci"][1],
            "ap": sens_res["ap_both"],
            "ap_ci_lo": sens_res["ap_both_ci"][0],
            "ap_ci_hi": sens_res["ap_both_ci"][1],
            "delta_auroc_vs_ref": sens_res["d_auc"],
            "delta_auroc_ci_lo": sens_res["d_auc_ci"][0],
            "delta_auroc_ci_hi": sens_res["d_auc_ci"][1],
            "delta_ap_vs_ref": sens_res["d_ap"],
            "delta_ap_ci_lo": sens_res["d_ap_ci"][0],
            "delta_ap_ci_hi": sens_res["d_ap_ci"][1],
            "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
        }
    ]

    with open("results/table_sensitivity_2018_2020.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sens_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT 3: UNRESTRICTED CALIPER NEGATIVES PERFORMANCE TABLE
    # ──────────────────────────────────────────────────────────────────────────
    t_unrest_rows = []
    for T in CFG["eval_cutoffs"]:
        for r in res_42["eval_unrest"][T]:
            t_unrest_rows.append({
                "cutoff": T,
                "negative_pool": "Unrestricted Concept Graph",
                "model": r["model"],
                "auroc": r["auc"],
                "auroc_ci_lo": r["auc_ci"][0],
                "auroc_ci_hi": r["auc_ci"][1],
                "ap": r["ap"],
                "ap_ci_lo": r["ap_ci"][0],
                "ap_ci_hi": r["ap_ci"][1],
                "delta_auroc_vs_ref": r["d_auc"],
                "delta_auroc_ci_lo": r["d_auc_ci"][0],
                "delta_auroc_ci_hi": r["d_auc_ci"][1],
                "shuffled_auroc": r["shuf_auc"],
                "shuffled_auroc_ci_lo": r["shuf_auc_ci"][0],
                "shuffled_auroc_ci_hi": r["shuf_auc_ci"][1],
                "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    with open("results/table_unrestricted_caliper_performance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t_unrest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t_unrest_rows)

    # Save restricted caliper performance as Table 4
    t_rest_rows = []
    for T in CFG["eval_cutoffs"]:
        for r in res_42["eval_rest"][T]:
            t_rest_rows.append({
                "cutoff": T,
                "negative_pool": "Restricted (Test-Window Concepts)",
                "model": r["model"],
                "auroc": r["auc"],
                "auroc_ci_lo": r["auc_ci"][0],
                "auroc_ci_hi": r["auc_ci"][1],
                "ap": r["ap"],
                "ap_ci_lo": r["ap_ci"][0],
                "ap_ci_hi": r["ap_ci"][1],
                "delta_auroc_vs_ref": r["d_auc"],
                "delta_auroc_ci_lo": r["d_auc_ci"][0],
                "delta_auroc_ci_hi": r["d_auc_ci"][1],
                "shuffled_auroc": r["shuf_auc"],
                "shuffled_auroc_ci_lo": r["shuf_auc_ci"][0],
                "shuffled_auroc_ci_hi": r["shuf_auc_ci"][1],
                "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    with open("results/table4_restricted_caliper_performance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t_rest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t_rest_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT 4: NATURAL-CANDIDATES CHECK (5,000 UNSEEN PAIRS PER CUTOFF)
    # ──────────────────────────────────────────────────────────────────────────
    nat_res = run_natural_candidates_check(res_42, extractor, sample_size=5000)
    nat_rows = []
    for c_res in nat_res:
        T = c_res["cutoff"]
        for m_name, m_metrics in c_res["models"].items():
            nat_rows.append({
                "cutoff": T,
                "n_sampled_pairs": c_res["sample_size"],
                "n_realized_positives": c_res["n_pos"],
                "base_rate_pct": c_res["base_rate"],
                "model": m_name,
                "auroc": m_metrics["auc"],
                "ap": m_metrics["ap"],
                "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    with open("results/table_natural_candidates_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(nat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(nat_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT 5 / TABLE 6: RQ2 FINAL TABLE (CITATION REGRESSIONS)
    # ──────────────────────────────────────────────────────────────────────────
    rq2_records = run_rq2_paper_regressions(
        res_42["pooled_first_papers"],
        primary_key="score_ref",
        primary_name="Reference Baseline Probability"
    )

    aggregation_rule_doc = "One row per unique first-realizing paper p; paper novelty = max predicted probability across realized novel pairs in p; outcome = within-year citation percentile rank; OLS with publication year fixed effects and paper-clustered SE."
    t6_rows = []
    for r in rq2_records:
        t6_rows.append({
            "predictor": r["predictor"],
            "role": r["role"],
            "beta": r["beta"],
            "clust_se": r["se"],
            "ci_lo": r["ci"][0],
            "ci_hi": r["ci"][1],
            "t_stat": r["t"],
            "p_val_adj": r["adj_p"],
            "p_val_raw": r["raw_p"],
            "n_papers": r["n"],
            "aggregation_rule": aggregation_rule_doc,
            "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE,
        })

    with open("results/table6_rq2_citation_regressions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(t6_rows[0].keys()))
        writer.writeheader()
        writer.writerows(t6_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # PRINT FORMATTED TABLES TO STDOUT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 136, flush=True)
    print("TABLE 5: MULTI-SEED REPLICATION (SEEDS 42, 43, 44): ΔAUROC AND ΔAP VS. REFERENCE BASELINE AT 2018 AND 2020", flush=True)
    print(f"Metadata: Git Commit {GIT_COMMIT} | Data Snapshot {DATA_SNAPSHOT_DATE}", flush=True)
    print("=" * 136, flush=True)
    hdr_t5 = f"{'Cutoff':<8} │ {'Model':<40} │ {'Metric':<8} │ {'Seed 42':>10} │ {'Seed 43':>10} │ {'Seed 44':>10} │ {'Mean ± SD':^18}"
    sep_t5 = "─" * 9 + "┼" + "─" * 42 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 20
    print(hdr_t5, flush=True)
    print(sep_t5, flush=True)

    for r in t5_rows:
        if r["model"] in ["Reference + Both Distances", "Reference Baseline + Momentum", "Reference + Momentum + Both Distances", "Hard Gate: AA Plausibility (Ablation)"]:
            mean_sd_auc = f"{r['d_auroc_mean']:+.4f} ± {r['d_auroc_sd']:.4f}"
            mean_sd_ap  = f"{r['d_ap_mean']:+.4f} ± {r['d_ap_sd']:.4f}"
            print(f"T={r['cutoff']:<6} │ {r['model']:<40} │ {'ΔAUROC':<8} │ {r['d_auroc_seed42']:>+10.4f} │ {r['d_auroc_seed43']:>+10.4f} │ {r['d_auroc_seed44']:>+10.4f} │ {mean_sd_auc:^18}", flush=True)
            print(f"{'':<8} │ {r['model']:<40} │ {'ΔAP':<8} │ {r['d_ap_seed42']:>+10.4f} │ {r['d_ap_seed43']:>+10.4f} │ {r['d_ap_seed44']:>+10.4f} │ {mean_sd_ap:^18}", flush=True)
            print(sep_t5, flush=True)

    print("\n" + "=" * 120, flush=True)
    print("TABLE: SENSITIVITY ANALYSIS (FIT ON T=2018, EVALUATED ON T=2020 FROZEN)", flush=True)
    print(f"Metadata: Git Commit {GIT_COMMIT} | Data Snapshot {DATA_SNAPSHOT_DATE}", flush=True)
    print("=" * 120, flush=True)
    hdr_s = f"{'Model':<36} │ {'AUROC [95% CI]':^24} │ {'ΔAUROC vs Ref [95% CI]':^28} │ {'AP [95% CI]':^24}"
    sep_s = "─" * 37 + "┼" + "─" * 26 + "┼" + "─" * 30 + "┼" + "─" * 26
    print(hdr_s, flush=True)
    print(sep_s, flush=True)
    for sr in sens_rows:
        auc_s = f"{sr['auroc']:.4f} [{sr['auroc_ci_lo']:.4f}, {sr['auroc_ci_hi']:.4f}]"
        ap_s  = f"{sr['ap']:.4f} [{sr['ap_ci_lo']:.4f}, {sr['ap_ci_hi']:.4f}]"
        d_auc_s = f"{sr['delta_auroc_vs_ref']:+.4f} [{sr['delta_auroc_ci_lo']:+.4f}, {sr['delta_auroc_ci_hi']:+.4f}]" if sr["model"] != "AA + Degree (Reference Baseline)" else "Reference"
        print(f"{sr['model']:<36} │ {auc_s:^24} │ {d_auc_s:^28} │ {ap_s:^24}", flush=True)
    print(sep_s, flush=True)

    print("\n" + "=" * 140, flush=True)
    print("TABLE: UNRESTRICTED CALIPER NEGATIVES PERFORMANCE ACROSS CUTOFFS", flush=True)
    print(f"Metadata: Git Commit {GIT_COMMIT} | Data Snapshot {DATA_SNAPSHOT_DATE}", flush=True)
    print("=" * 140, flush=True)
    hdr_u = f"{'Cutoff':<8} │ {'Model':<40} │ {'AUROC [95% CI]':^24} │ {'AP [95% CI]':^24} │ {'ΔAUROC vs Ref [95% CI]':^28} │ {'Shuffled AUROC':^14}"
    sep_u = "─" * 9 + "┼" + "─" * 42 + "┼" + "─" * 26 + "┼" + "─" * 26 + "┼" + "─" * 30 + "┼" + "─" * 16
    print(hdr_u, flush=True)
    print(sep_u, flush=True)
    for ur in t_unrest_rows:
        auc_s = f"{ur['auroc']:.4f} [{ur['auroc_ci_lo']:.4f}, {ur['auroc_ci_hi']:.4f}]"
        ap_s  = f"{ur['ap']:.4f} [{ur['ap_ci_lo']:.4f}, {ur['ap_ci_hi']:.4f}]"
        d_auc_s = f"{ur['delta_auroc_vs_ref']:+.4f} [{ur['delta_auroc_ci_lo']:+.4f}, {ur['delta_auroc_ci_hi']:+.4f}]"
        shuf_s = f"{ur['shuffled_auroc']:.4f}"
        print(f"T={ur['cutoff']:<6} │ {ur['model']:<40} │ {auc_s:^24} │ {ap_s:^24} │ {d_auc_s:^28} │ {shuf_s:^14}", flush=True)
        if ur["model"] == "Hard Gate: Embedding Dist <= tau (Abl.)":
            print(sep_u, flush=True)

    print("\n" + "=" * 100, flush=True)
    print("TABLE: NATURAL-CANDIDATES CHECK (5,000 RANDOM UNSEEN PAIRS PER CUTOFF, NO MATCHING)", flush=True)
    print(f"Metadata: Git Commit {GIT_COMMIT} | Data Snapshot {DATA_SNAPSHOT_DATE}", flush=True)
    print("=" * 100, flush=True)
    hdr_n = f"{'Cutoff':<8} │ {'Realized Pos (N)':>18} │ {'Base Rate':>12} │ {'Model':<36} │ {'AUROC':>10} │ {'AP':>10}"
    sep_n = "─" * 9 + "┼" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 38 + "┼" + "─" * 12 + "┼" + "─" * 12
    print(hdr_n, flush=True)
    print(sep_n, flush=True)
    for nr in nat_rows:
        pos_str = f"{nr['n_realized_positives']} / {nr['n_sampled_pairs']}"
        br_str = f"{nr['base_rate_pct']:.2f}%"
        print(f"T={nr['cutoff']:<6} │ {pos_str:>18} │ {br_str:>12} │ {nr['model']:<36} │ {nr['auroc']:>10.4f} │ {nr['ap']:>10.4f}", flush=True)
        if nr["model"] == "Ref + Both Distances":
            print(sep_n, flush=True)

    print("\n" + "=" * 136, flush=True)
    print("TABLE 6: RQ2 WITHIN-YEAR CITATION PERCENTILE RANK REGRESSED ON NOVELTY PROBABILITIES", flush=True)
    print(f"Metadata: Git Commit {GIT_COMMIT} | Data Snapshot {DATA_SNAPSHOT_DATE}", flush=True)
    print("Per-Paper Aggregation Rule: 1 row per unique first-realizing paper, novelty = max prob across pairs in paper,", flush=True)
    print("outcome = within-year citation percentile rank, publication year fixed effects, clustered SE by paper.", flush=True)
    print("=" * 136, flush=True)
    hdr_6 = f"{'Predictor Variable':<52} │ {'Role':<12} │ {'Beta':>10} │ {'Clust SE':>10} │ {'95% CI (Adj.)':^24} │ {'t-stat':>8} │ {'Adj. p':>10} │ {'N':>6}"
    sep_6 = "─" * 53 + "┼" + "─" * 14 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 26 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 8
    print(hdr_6, flush=True)
    print(sep_6, flush=True)
    for r6 in t6_rows:
        ci_s = f"[{r6['ci_lo']:+.4f}, {r6['ci_hi']:+.4f}]"
        p_s = f"{r6['p_val_adj']:.4f}" if r6['p_val_adj'] >= 0.0001 else "<0.0001"
        print(f"{r6['predictor']:<52} │ {r6['role']:<12} │ {r6['beta']:>+10.4f} │ {r6['clust_se']:>10.4f} │ {ci_s:^24} │ {r6['t_stat']:>+8.2f} │ {p_s:>10} │ {r6['n_papers']:>6,d}", flush=True)
    print(sep_6, flush=True)

    print(f"\nAll operations completed and tables saved to results/*.csv in {time.time() - t_start:.2f}s.", flush=True)

if __name__ == "__main__":
    main()
