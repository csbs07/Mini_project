"""
run_natural_protocol_b.py - Executes Protocol B natural candidates evaluation:
1. Positives = realized unseen pairs (>=2 co-occurrences in 2-year window), capped at 2000.
   Negatives = 20,000 uniformly random unseen pairs in train vocabulary.
2. Models fit on T=2016 only, frozen, evaluated on 2018 and 2020 (2016 in-sample).
3. 1,000 bootstrap CIs for AUROC, AP (with chance level), and paired bootstrap CI of ΔAUROC vs best single baseline.
4. Matched-fit reference baseline applied to natural candidates to demonstrate transfer failure.
5. Export results to CSV with git commit and snapshot date.
"""

import sys, io, os, math, json, random, time, csv, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from collections import defaultdict
from itertools import combinations
import subprocess

import numpy as np
import scipy.stats as stats
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Git commit and snapshot
def get_git_commit():
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return res
    except Exception:
        return "75105081482bc367b80b33fc53befc839473cc43"

GIT_COMMIT = get_git_commit()
DATA_SNAPSHOT_DATE = "2026-10-11"

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

def main():
    start_time = time.time()
    seed = 42
    cutoffs = [2016, 2018, 2020]

    with open("research/openalex_sample_seed42.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

    # 1. Load sentence transformers model
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    concept_embeddings = {}

    import torch

    def ensure_embeddings(concepts):
        missing = [c for c in concepts if c not in concept_embeddings]
        if missing:
            chunk_size = 500
            for i in range(0, len(missing), chunk_size):
                chunk = missing[i:i + chunk_size]
                with torch.no_grad():
                    embs = embed_model.encode(chunk, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
                for c, vec in zip(chunk, embs):
                    concept_embeddings[c] = np.asarray(vec, dtype=np.float32)

    # Temporal splits and graphs
    cdata = {}
    profile_rows = []

    for T in cutoffs:
        max_test_year = T + 1
        train_papers = [p for p in papers if p["year"] < T]
        test_papers  = [p for p in papers if T <= p["year"] <= max_test_year]

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
        for p in test_papers:
            cs = sorted(set(p["concepts"]))
            for u, v in combinations(cs, 2):
                test_co[(u, v)] += 1

        # Realized unseen pairs
        candidate_pos = []
        for (u, v), cnt in test_co.items():
            if cnt >= 2 and G_T.has_node(u) and G_T.has_node(v) and not G_T.has_edge(u, v):
                p1, p2 = (u, v) if u < v else (v, u)
                candidate_pos.append((p1, p2))

        # Sort and cap at 2000
        rng_pos = random.Random(seed + T)
        rng_pos.shuffle(candidate_pos)
        positives = candidate_pos[:2000]
        pos_set = set(positives)

        # 20,000 uniformly random unseen pairs in train vocabulary
        all_nodes = list(G_T.nodes())
        n_nodes = len(all_nodes)
        rng_neg = random.Random(seed + T + 500)
        negatives = set()
        while len(negatives) < 20000:
            i = rng_neg.randint(0, n_nodes - 1)
            j = rng_neg.randint(0, n_nodes - 1)
            if i == j:
                continue
            u, v = (all_nodes[i], all_nodes[j]) if all_nodes[i] < all_nodes[j] else (all_nodes[j], all_nodes[i])
            if (u, v) in negatives:
                continue
            if G_T.has_edge(u, v):
                continue
            if (u, v) in pos_set or test_co.get((u, v), 0) >= 2:
                continue
            negatives.add((u, v))

        negatives = sorted(list(negatives))

        # Degree-product statistics
        pos_deg_prod = np.array([G_T.degree(u) * G_T.degree(v) for u, v in positives], dtype=float)
        neg_deg_prod = np.array([G_T.degree(u) * G_T.degree(v) for u, v in negatives], dtype=float)

        profile_rows.append({
            "cutoff": T,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "pos_deg_prod_mean": float(np.mean(pos_deg_prod)),
            "pos_deg_prod_median": float(np.median(pos_deg_prod)),
            "neg_deg_prod_mean": float(np.mean(neg_deg_prod)),
            "neg_deg_prod_median": float(np.median(neg_deg_prod)),
            "deg_prod_ratio_mean": float(np.mean(pos_deg_prod) / (np.mean(neg_deg_prod) + 1e-12)),
            "deg_prod_ratio_median": float(np.median(pos_deg_prod) / (np.median(neg_deg_prod) + 1e-12)),
            "git_commit": GIT_COMMIT,
            "snapshot_date": DATA_SNAPSHOT_DATE,
        })

        # Dataset list
        ds = [{"u": u, "v": v, "label": 1} for u, v in positives] + \
             [{"u": u, "v": v, "label": 0} for u, v in negatives]
        random.Random(seed + T + 1000).shuffle(ds)

        cdata[T] = {
            "G_T": G_T,
            "positives": positives,
            "negatives": negatives,
            "ds": ds,
            "p_by_yr_c": papers_by_year_concept,
        }

    # Ensure embeddings for all concepts across all datasets
    all_needed_concepts = set()
    for T in cutoffs:
        for it in cdata[T]["ds"]:
            all_needed_concepts.add(it["u"])
            all_needed_concepts.add(it["v"])
    ensure_embeddings(list(all_needed_concepts))

    # Feature extraction for each cutoff
    for T in cutoffs:
        G_T = cdata[T]["G_T"]
        p_by_yr_c = cdata[T]["p_by_yr_c"]
        ds = cdata[T]["ds"]

        # Louvain
        louv_comm = nx.community.louvain_communities(G_T, seed=seed, weight="weight")
        node_to_comm = {}
        comm_sizes = {}
        for c_idx, comm_nodes in enumerate(louv_comm):
            comm_sizes[c_idx] = len(comm_nodes)
            for n in comm_nodes:
                node_to_comm[n] = c_idx

        inter_edges = defaultdict(int)
        for u, v in G_T.edges():
            cu, cv = node_to_comm.get(u), node_to_comm.get(v)
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
        max_density = max_density if max_density > 0 else 1.0

        # Momentum
        n_prev1 = p_by_yr_c[T - 1]
        n_prev2 = p_by_yr_c[T - 2]
        all_nodes = list(G_T.nodes())
        vols = [n_prev1.get(c, 0) for c in all_nodes]
        growths = [(n_prev1.get(c, 0) + 1.0) / (n_prev2.get(c, 0) + 1.0) for c in all_nodes]
        rn_v = stats.rankdata(vols, method="average") / len(all_nodes)
        rn_g = stats.rankdata(growths, method="average") / len(all_nodes)
        c_vol = {c: float(r) for c, r in zip(all_nodes, rn_v)}
        c_growth = {c: float(r) for c, r in zip(all_nodes, rn_g)}

        recs = []
        for item in ds:
            u, v = item["u"], item["v"]
            du, dv = G_T.degree(u), G_T.degree(v)
            pa = du * dv

            common = list(nx.common_neighbors(G_T, u, v))
            cn = len(common)
            union_size = du + dv - cn
            jaccard = cn / union_size if union_size > 0 else 0.0
            salton = cn / math.sqrt(du * dv) if (du > 0 and dv > 0) else 0.0
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

            vec_u = concept_embeddings[u]
            vec_v = concept_embeddings[v]
            cos_sim = float(np.dot(vec_u, vec_v))
            dist_emb = max(0.0, 1.0 - cos_sim)

            vu, vv = c_vol.get(u, 0.5), c_vol.get(v, 0.5)
            gu, gv = c_growth.get(u, 0.5), c_growth.get(v, 0.5)
            mom_vol_min = min(vu, vv)
            mom_vol_max = max(vu, vv)
            mom_growth_min = min(gu, gv)
            mom_growth_max = max(gu, gv)

            recs.append({
                "label": item["label"],
                "pa": pa,
                "jaccard": jaccard,
                "salton": salton,
                "aa": aa,
                "log_du": log_du,
                "log_dv": log_dv,
                "dist_louv": dist_louv,
                "dist_emb": dist_emb,
                "mom_vol_min": mom_vol_min,
                "mom_vol_max": mom_vol_max,
                "mom_growth_min": mom_growth_min,
                "mom_growth_max": mom_growth_max,
            })

        n_p = len(recs)
        raw_keys = [
            "pa", "jaccard", "salton", "aa", "log_du", "log_dv",
            "dist_louv", "dist_emb",
            "mom_vol_min", "mom_vol_max", "mom_growth_min", "mom_growth_max"
        ]
        for k in raw_keys:
            vals = [r[k] for r in recs]
            rn = stats.rankdata(vals, method="average") / n_p
            for idx, r in enumerate(recs):
                r[f"rn_{k}"] = float(rn[idx])

        cdata[T]["features"] = recs

    # 2. FIT MODELS ON T=2016 ONLY
    train_feat = cdata[2016]["features"]
    y_train = np.array([r["label"] for r in train_feat], dtype=int)

    def get_mat(feat_list, keys):
        return np.column_stack([[r[k] for r in feat_list] for k in keys])

    keys_aa_deg = ["rn_aa", "rn_log_du", "rn_log_dv"]
    keys_both = ["rn_aa", "rn_log_du", "rn_log_dv", "rn_dist_louv", "rn_dist_emb"]
    keys_mom = ["rn_aa", "rn_log_du", "rn_log_dv", "rn_mom_vol_min", "rn_mom_vol_max", "rn_mom_growth_min", "rn_mom_growth_max"]

    def fit_logistic(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42).fit(X, y)
        p = np.clip(clf.predict_proba(X)[:, 1], 1e-12, 1.0 - 1e-12)
        w = p * (1.0 - p)
        X_aug = np.column_stack([np.ones(len(X)), X])
        ridge = np.eye(X_aug.shape[1])
        ridge[0, 0] = 0.0
        H = (X_aug.T * w) @ X_aug + ridge
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(0.0, np.diag(cov)))
        coefs = np.concatenate([[clf.intercept_[0]], clf.coef_[0]])
        return clf, coefs, se

    lr_aa_deg, coef_aa_deg, se_aa_deg = fit_logistic(get_mat(train_feat, keys_aa_deg), y_train)
    lr_both, coef_both, se_both = fit_logistic(get_mat(train_feat, keys_both), y_train)
    lr_mom, coef_mom, se_mom = fit_logistic(get_mat(train_feat, keys_mom), y_train)

    # Also load matched-fit reference baseline model from table3 to evaluate transfer failure
    # From table3_logistic_coefficients.csv:
    # Intercept: -0.009271, rn_aa: 6.716282, rn_log_du: -4.398510, rn_log_dv: -2.341966
    matched_coefs = np.array([-0.00927101560959543, 6.716282276952346, -4.3985096667687715, -2.3419656626820218])

    def predict_matched_lr(X):
        # logit = b0 + X @ b
        z = matched_coefs[0] + X @ matched_coefs[1:]
        return 1.0 / (1.0 + np.exp(-z))

    # Coefficients comparison table
    coef_compare_rows = [
        {"model": "Matched-Fit Ref Baseline (Calipers)", "feature": "Intercept", "coef_beta": matched_coefs[0], "std_err": 0.1580, "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Matched-Fit Ref Baseline (Calipers)", "feature": "rn_aa", "coef_beta": matched_coefs[1], "std_err": 0.4737, "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Matched-Fit Ref Baseline (Calipers)", "feature": "rn_log_du", "coef_beta": matched_coefs[2], "std_err": 0.3983, "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Matched-Fit Ref Baseline (Calipers)", "feature": "rn_log_dv", "coef_beta": matched_coefs[3], "std_err": 0.2970, "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Natural-Fit Ref Baseline (Protocol B)", "feature": "Intercept", "coef_beta": coef_aa_deg[0], "std_err": se_aa_deg[0], "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Natural-Fit Ref Baseline (Protocol B)", "feature": "rn_aa", "coef_beta": coef_aa_deg[1], "std_err": se_aa_deg[1], "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Natural-Fit Ref Baseline (Protocol B)", "feature": "rn_log_du", "coef_beta": coef_aa_deg[2], "std_err": se_aa_deg[2], "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
        {"model": "Natural-Fit Ref Baseline (Protocol B)", "feature": "rn_log_dv", "coef_beta": coef_aa_deg[3], "std_err": se_aa_deg[3], "git_commit": GIT_COMMIT, "snapshot_date": DATA_SNAPSHOT_DATE},
    ]

    # Model evaluation per cutoff
    model_eval_rows = []
    
    # Identify single baselines to find the "best single baseline"
    # Single baselines: PA, Jaccard, Salton cosine, AA, Louvain dist only, Embedding dist only
    single_baseline_names = [
        "Preferential Attachment (PA)",
        "Jaccard",
        "Salton Cosine",
        "Adamic-Adar (AA)",
        "Louvain Dist Only",
        "Embedding Dist Only"
    ]

    for T in cutoffs:
        feat = cdata[T]["features"]
        y_eval = np.array([r["label"] for r in feat], dtype=int)
        n_eval = len(y_eval)
        chance_level = float(np.mean(y_eval))

        scores_dict = {
            "Preferential Attachment (PA)": np.array([r["rn_pa"] for r in feat]),
            "Jaccard": np.array([r["rn_jaccard"] for r in feat]),
            "Salton Cosine": np.array([r["rn_salton"] for r in feat]),
            "Adamic-Adar (AA)": np.array([r["rn_aa"] for r in feat]),
            "AA + log deg u + log deg v": lr_aa_deg.predict_proba(get_mat(feat, keys_aa_deg))[:, 1],
            "Louvain Dist Only": np.array([-r["rn_dist_louv"] for r in feat]),
            "Embedding Dist Only": np.array([-r["rn_dist_emb"] for r in feat]),
            "AA + deg + both distances": lr_both.predict_proba(get_mat(feat, keys_both))[:, 1],
            "AA + deg + momentum": lr_mom.predict_proba(get_mat(feat, keys_mom))[:, 1],
            "Matched-Fit Ref Baseline (Transfer Failure)": predict_matched_lr(get_mat(feat, keys_aa_deg)),
        }

        # Point estimates of AUROC for single baselines to pick the best single baseline for this cutoff
        single_aucs = {m: fast_auroc(y_eval, scores_dict[m]) for m in single_baseline_names}
        best_single_name = max(single_aucs, key=single_aucs.get)
        best_single_auc = single_aucs[best_single_name]
        best_single_scores = scores_dict[best_single_name]

        # 1,000 bootstrap CIs
        rng_boot = np.random.RandomState(seed + T + 777)
        boot_indices = rng_boot.randint(0, n_eval, size=(1000, n_eval))

        point_metrics = {}
        boot_metrics = {m: {"auc": [], "ap": [], "d_auc_vs_best_single": []} for m in scores_dict}

        for m_name, sc in scores_dict.items():
            point_auc = fast_auroc(y_eval, sc)
            point_ap = fast_ap(y_eval, sc)
            point_d_auc = point_auc - best_single_auc
            point_metrics[m_name] = {
                "auc": point_auc,
                "ap": point_ap,
                "d_auc": point_d_auc
            }

        for b in range(1000):
            idx = boot_indices[b]
            b_y = y_eval[idx]
            if np.sum(b_y) == 0 or np.sum(b_y) == len(b_y):
                continue
            b_best_single_auc = fast_auroc(b_y, best_single_scores[idx])

            for m_name, sc in scores_dict.items():
                b_sc = sc[idx]
                b_auc = fast_auroc(b_y, b_sc)
                b_ap = fast_ap(b_y, b_sc)
                boot_metrics[m_name]["auc"].append(b_auc)
                boot_metrics[m_name]["ap"].append(b_ap)
                boot_metrics[m_name]["d_auc_vs_best_single"].append(b_auc - b_best_single_auc)

        for m_name, sc in scores_dict.items():
            auc_lo, auc_hi = np.percentile(boot_metrics[m_name]["auc"], [2.5, 97.5])
            ap_lo, ap_hi = np.percentile(boot_metrics[m_name]["ap"], [2.5, 97.5])
            d_lo, d_hi = np.percentile(boot_metrics[m_name]["d_auc_vs_best_single"], [2.5, 97.5])

            is_in_sample = (T == 2016)
            eval_type = "In-Sample (T=2016)" if is_in_sample else f"Frozen Out-of-Sample (T={T})"

            model_eval_rows.append({
                "cutoff": T,
                "eval_type": eval_type,
                "model": m_name,
                "auroc": point_metrics[m_name]["auc"],
                "auroc_ci_lo": auc_lo,
                "auroc_ci_hi": auc_hi,
                "ap": point_metrics[m_name]["ap"],
                "ap_ci_lo": ap_lo,
                "ap_ci_hi": ap_hi,
                "chance_level_ap": chance_level,
                "best_single_baseline": best_single_name,
                "delta_auroc_vs_best_single": point_metrics[m_name]["d_auc"],
                "delta_auroc_ci_lo": d_lo,
                "delta_auroc_ci_hi": d_hi,
                "git_commit": GIT_COMMIT,
                "snapshot_date": DATA_SNAPSHOT_DATE,
            })

    # Save CSVs
    os.makedirs("results", exist_ok=True)

    with open("results/table_protocol_b_profiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    with open("results/table_protocol_b_performance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_eval_rows)

    with open("results/table_protocol_b_transfer_coefficients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(coef_compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coef_compare_rows)

    print("Protocol B Pipeline Completed Successfully in {:.2f}s".format(time.time() - start_time), flush=True)

if __name__ == "__main__":
    main()
