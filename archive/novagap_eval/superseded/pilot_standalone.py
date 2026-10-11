"""
pilot_standalone.py  —  NovaGap-Lite Hypothesis Verification
=============================================================
Pure Python + stdlib + networkx + requests ONLY.
No numpy, no scikit-learn, no OpenBLAS, no torch.

Run:
    py -3 research/pilot_standalone.py

What this does:
  1. Pulls ~1 600 CS papers from OpenAlex (free, no API key)
  2. Normalises concepts & builds temporal graph G_T (cutoff 2018)
  3. Samples degree-matched positive / negative absent-pairs
  4. Scores: Preferential Attachment, Adamic-Adar, Uzzi Atypicality,
            Atomic-Proxy, and NovaGap-Lite (Gated = Atypicality x Plausibility)
  5. Reports AUROC, Avg Precision, P@50 for every method  (RQ1)
  6. Spearman-rho of novelty vs citations for realized pairs (RQ2)
  7. Gate ablation: does the gate reduce mirage candidates?  (RQ3)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import re, math, json, time, random
import requests
import networkx as nx
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    from_year    = 2013,
    to_year      = 2021,
    per_page     = 200,
    max_pages    = 8,        # 8 × 200 = 1 600 papers — fast pilot
    cutoff_T     = 2018,
    eval_window  = 3,        # evaluate 2018-2021
    n_pos        = 300,
    n_neg        = 300,
    max_path_len = 3,        # plausibility gate: max hop distance
    min_aa       = 0.05,     # plausibility gate: min Adamic-Adar
    cache        = Path(__file__).parent / "openalex_cache.json",
    seed         = 42,
)

ALIAS = {
    "neural network":          "deep learning",
    "neural networks":         "deep learning",
    "convolutional neural":    "deep learning",
    "convolutional network":   "deep learning",
    "natural language":        "natural language processing",
    "transformer":             "transformer model",
    "attention mechanism":     "transformer model",
    "self-attention":          "transformer model",
    "large language model":    "large language models",
    "knowledge graph":         "knowledge graphs",
    "graph neural network":    "graph neural networks",
    "federated learning":      "federated learning",
    "transfer learning":       "transfer learning",
}

# ──────────────────────────────────────────────────────────────────────────────
# STATS  (pure Python — no numpy needed)
# ──────────────────────────────────────────────────────────────────────────────

def auroc(y_true, scores):
    """Area under ROC curve — O(n log n) sort-based."""
    paired = sorted(zip(scores, y_true), reverse=True)
    n_pos  = sum(y_true)
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = auc = prev_tp = 0
    prev_score = None
    for score, label in paired:
        if score != prev_score and prev_score is not None:
            auc += (fp - 0) * (tp + prev_tp) / 2  # trapezoid
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
        prev_tp    = tp
    auc += fp * (tp + prev_tp) / 2
    return auc / (n_pos * n_neg)


def avg_precision(y_true, scores):
    """Average Precision (area under Precision-Recall curve)."""
    paired   = sorted(zip(scores, y_true), reverse=True)
    tp = 0
    ap = 0.0
    for i, (_, label) in enumerate(paired, 1):
        if label == 1:
            tp += 1
            ap += tp / i
    n_pos = sum(y_true)
    return ap / n_pos if n_pos else 0.0


def precision_at_k(y_true, scores, k=50):
    paired = sorted(zip(scores, y_true), reverse=True)[:k]
    return sum(label for _, label in paired) / len(paired) if paired else 0.0


def spearman_rho(xs, ys):
    """Spearman rank correlation — pure Python."""
    n = len(xs)
    if n < 3:
        return 0.0
    def ranks(lst):
        order = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and lst[order[j]] == lst[order[j+1]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    d2 = sum((rx[i] - ry[i])**2 for i in range(n))
    return 1 - 6 * d2 / (n * (n*n - 1))

# ──────────────────────────────────────────────────────────────────────────────
# DATA INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def normalize(raw: str) -> str:
    c = raw.lower().strip()
    c = re.sub(r"[^\w\s-]", "", c)
    c = re.sub(r"\s+", " ", c)
    for frag, canon in ALIAS.items():
        if frag in c:
            return canon
    return c


def fetch_papers(cfg: dict) -> list:
    if cfg["cache"].exists():
        print(f"[cache] Loading from {cfg['cache']}")
        with open(cfg["cache"]) as f:
            return json.load(f)

    papers   = []
    base_url = "https://api.openalex.org/works"
    cs_id    = "c41008148"      # Computer Science in OpenAlex

    print("[OpenAlex] Fetching papers…")
    for page in range(1, cfg["max_pages"] + 1):
        params = dict(
            filter   = f"concepts.id:{cs_id},"
                       f"publication_year:{cfg['from_year']}-{cfg['to_year']}",
            per_page = cfg["per_page"],
            page     = page,
            select   = "id,title,publication_year,concepts,cited_by_count",
        )
        try:
            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} on page {page} — stopping."); break
            items = r.json().get("results", [])
            if not items:
                break
            for item in items:
                concepts = list({
                    normalize(c["display_name"])
                    for c in item.get("concepts", [])
                    if c.get("score", 0) > 0.3
                    and len(c.get("display_name", "")) > 2
                })
                if len(concepts) >= 2:
                    papers.append(dict(
                        year     = item["publication_year"],
                        citations= item.get("cited_by_count", 0),
                        concepts = concepts,
                    ))
            print(f"  page {page}/{cfg['max_pages']}  → cumulative {len(papers)} papers")
            time.sleep(0.25)
        except Exception as e:
            print(f"  error page {page}: {e}"); break

    with open(cfg["cache"], "w") as f:
        json.dump(papers, f)
    return papers

# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL GRAPH BUILD
# ──────────────────────────────────────────────────────────────────────────────

def build_split(papers, cutoff_T, eval_window):
    """
    Returns:
      G_train      — nx.Graph co-occurrence graph (year < cutoff_T)
      test_co      — {(u,v): count}  in [cutoff_T, cutoff_T+eval_window]
      test_cites   — {(u,v): max_citations}  same window
    """
    G          = nx.Graph()
    test_co    = defaultdict(int)
    test_cites = defaultdict(int)
    max_year   = cutoff_T + eval_window

    for p in papers:
        yr  = p["year"]
        cs  = list(set(p["concepts"]))
        cit = p["citations"]

        if yr < cutoff_T:
            for c in cs:
                if c not in G:
                    G.add_node(c, paper_count=0)
                G.nodes[c]["paper_count"] += 1
            for u, v in combinations(cs, 2):
                a, b = (u, v) if u < v else (v, u)
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

        elif cutoff_T <= yr <= max_year:
            for u, v in combinations(cs, 2):
                pair = (u, v) if u < v else (v, u)
                test_co[pair]    += 1
                test_cites[pair]  = max(test_cites[pair], cit)

    return G, dict(test_co), dict(test_cites)

# ──────────────────────────────────────────────────────────────────────────────
# METRICS  (pure Python, no numpy)
# ──────────────────────────────────────────────────────────────────────────────

def atypicality(u, v, G):
    """Uzzi-style z-score, inverted so HIGH = more atypical."""
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    total_w = sum(d["weight"] for _, _, d in G.edges(data=True))
    if total_w == 0:
        return 0.0
    d_u     = sum(d["weight"] for _, _, d in G.edges(u, data=True))
    d_v     = sum(d["weight"] for _, _, d in G.edges(v, data=True))
    expected= (d_u * d_v) / (2.0 * total_w)
    actual  = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
    std     = math.sqrt(expected) if expected > 0 else 1.0
    return -(actual - expected) / std          # invert: rare = positive


def plausibility(u, v, G):
    """Returns (shortest_path_len, adamic_adar). path=999 if disconnected."""
    if not G.has_node(u) or not G.has_node(v):
        return 999, 0.0
    try:
        pl = nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        pl = 999
    aa = sum(
        1.0 / math.log(G.degree(w))
        for w in nx.common_neighbors(G, u, v)
        if G.degree(w) > 1
    )
    return pl, aa


def gated_novelty(u, v, G, max_pl, min_aa):
    atyp    = atypicality(u, v, G)
    pl, aa  = plausibility(u, v, G)
    ok      = (pl <= max_pl) or (aa >= min_aa)
    return atyp if ok else 0.0, pl, aa, ok

# ──────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def sample_pairs(G, test_co, test_cites, n_pos, n_neg, seed):
    rng   = random.Random(seed)
    nodes = list(G.nodes())

    # Positives: absent in G, co-occur >= 2 in test
    positives = [
        (u, v) for (u, v), cnt in test_co.items()
        if cnt >= 2 and G.has_node(u) and G.has_node(v)
        and not G.has_edge(u, v)
    ]
    rng.shuffle(positives)
    positives = positives[:n_pos]

    # Negatives: absent in G AND absent in test window, degree-matched
    negatives = []
    max_try   = n_neg * 300
    attempts  = 0
    neg_set   = set()

    while len(negatives) < len(positives) and attempts < max_try:
        attempts += 1
        ref_u, ref_v = positives[attempts % len(positives)]
        deg_u = G.degree(ref_u)
        deg_v = G.degree(ref_v)

        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u == v:
            continue
        tol_u = max(2, int(0.25 * deg_u))
        tol_v = max(2, int(0.25 * deg_v))
        if abs(G.degree(u) - deg_u) > tol_u:
            continue
        if abs(G.degree(v) - deg_v) > tol_v:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in neg_set:
            continue
        if not G.has_edge(u, v) and test_co.get(pair, 0) == 0:
            negatives.append(pair)
            neg_set.add(pair)

    dataset = (
        [dict(u=u, v=v, label=1,
              cites=test_cites.get((u,v) if u<v else (v,u), 0))
         for u, v in positives]
        +
        [dict(u=u, v=v, label=0, cites=0)
         for u, v in negatives]
    )
    rng.shuffle(dataset)
    return dataset

# ──────────────────────────────────────────────────────────────────────────────
# SCORING ALL BASELINES
# ──────────────────────────────────────────────────────────────────────────────

def score_all(dataset, G, max_pl, min_aa):
    print("  Scoring pairs…", end="", flush=True)
    scored = []
    for i, row in enumerate(dataset):
        u, v = row["u"], row["v"]

        # Preferential attachment
        pa = G.degree(u) * G.degree(v)

        # Adamic-Adar alone
        pl, aa = plausibility(u, v, G)

        # Uzzi atypicality alone
        atyp = atypicality(u, v, G)

        # Atomic-overlap proxy: fraction of concepts individually present
        # (1 = both absent = proxy-novel; 0 = both present = proxy-unoriginal)
        present      = sum(1 for c in [u, v] if G.has_node(c))
        atomic_proxy = 1.0 - present / 2.0

        # NovaGap-Lite gated
        gated, pl2, aa2, is_plausible = gated_novelty(u, v, G, max_pl, min_aa)

        scored.append({**row,
            "s_pa":      pa,
            "s_aa":      aa,
            "s_atyp":    atyp,
            "s_atomic":  atomic_proxy,
            "s_gated":   gated,
            "plausible": is_plausible,
        })
        if (i + 1) % 100 == 0:
            print(f" {i+1}", end="", flush=True)
    print()
    return scored

# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def banner(txt):
    line = "═" * 62
    print(f"\n{line}\n  {txt}\n{line}")


def show_rq1(scored):
    banner("RQ1  ·  Link-prediction accuracy (materialization)")
    y   = [d["label"] for d in scored]
    methods = {
        "Preferential Attachment": [d["s_pa"]     for d in scored],
        "Adamic-Adar Alone":       [d["s_aa"]     for d in scored],
        "Atypicality Alone":       [d["s_atyp"]   for d in scored],
        "Atomic Overlap Proxy":    [d["s_atomic"]  for d in scored],
        "NovaGap-Lite (Gated)":    [d["s_gated"]  for d in scored],
    }
    W = 26
    print(f"\n{'Method':<{W}} │ {'AUROC':>7} │ {'Avg Prec':>9} │ {'P@50':>6}")
    print("─" * (W + 33))
    results = {}
    for name, preds in methods.items():
        au = auroc(y, preds)
        ap = avg_precision(y, preds)
        p50= precision_at_k(y, preds, k=50)
        marker = " ◄" if "Gated" in name else ""
        print(f"{name:<{W}} │ {au:>7.4f} │ {ap:>9.4f} │ {p50:>6.4f}{marker}")
        results[name] = dict(auroc=au, ap=ap, p50=p50)
    return results


def show_rq2(scored):
    banner("RQ2  ·  Citation association for high-novelty realized pairs")
    realized = [d for d in scored if d["label"] == 1]
    if len(realized) < 4:
        print("  Too few realized pairs."); return {}
    gs = [d["s_gated"] for d in realized]
    cs = [d["cites"]   for d in realized]
    rho = spearman_rho(gs, cs)
    med_g  = sorted(gs)[len(gs)//2]
    hi_c   = [d["cites"] for d in realized if d["s_gated"] >= med_g]
    lo_c   = [d["cites"] for d in realized if d["s_gated"] <  med_g]
    med_hi = sorted(hi_c)[len(hi_c)//2] if hi_c else 0
    med_lo = sorted(lo_c)[len(lo_c)//2] if lo_c else 0
    print(f"  Realized pairs           : {len(realized)}")
    print(f"  Spearman ρ (novelty~cites): {rho:+.4f}")
    print(f"  High-novelty median cites : {med_hi}")
    print(f"  Low-novelty  median cites : {med_lo}")
    if rho > 0:
        print("  ✓ RQ2 SUPPORTED — higher novelty score correlates with more citations.")
    else:
        print("  ✗ RQ2 not supported on this sample (null result).")
    return dict(rho=rho)


def show_rq3(scored):
    banner("RQ3  ·  Gate ablation — does the plausibility gate reduce mirages?")
    threshold  = sorted(d["s_atyp"] for d in scored)[len(scored)//2]
    high_atyp  = [d for d in scored if d["s_atyp"] >= threshold]
    after_gate = [d for d in high_atyp if d["plausible"]]
    removed    = len(high_atyp) - len(after_gate)

    prec_raw  = sum(d["label"] for d in high_atyp)  / len(high_atyp)  if high_atyp  else 0
    prec_gate = sum(d["label"] for d in after_gate) / len(after_gate) if after_gate else 0

    print(f"  High-atypicality pairs (≥ median)  : {len(high_atyp)}")
    print(f"  After plausibility gate             : {len(after_gate)}")
    print(f"  Mirages removed by gate             : {removed}")
    print(f"  Precision without gate              : {prec_raw:.4f}")
    print(f"  Precision with gate                 : {prec_gate:.4f}")
    if prec_gate > prec_raw:
        print("  ✓ RQ3 SUPPORTED — gate improves precision, reduces mirage candidates.")
    else:
        print("  ✗ RQ3 not supported — try adjusting max_path_len / min_aa.")
    return dict(prec_raw=prec_raw, prec_gate=prec_gate)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    banner("NovaGap-Lite  ·  Hypothesis Verification Pilot")

    # 1. Data
    papers = fetch_papers(CFG)
    print(f"\n✓ Papers loaded   : {len(papers)}")
    from collections import Counter
    all_c = [c for p in papers for c in p["concepts"]]
    top10 = Counter(all_c).most_common(10)
    print(f"✓ Concept tokens  : {len(all_c)}")
    print(f"✓ Top-10 concepts : {', '.join(f'{c}({n})' for c,n in top10)}")

    # 2. Temporal split
    T = CFG["cutoff_T"]
    print(f"\n⚙  Building G_train (papers before {T})…")
    G, test_co, test_cites = build_split(papers, T, CFG["eval_window"])
    n_pos_cands = sum(1 for v in test_co.values() if v >= 2)
    print(f"✓ G_train nodes   : {G.number_of_nodes()}")
    print(f"✓ G_train edges   : {G.number_of_edges()}")
    print(f"✓ Test-window pairs (count≥2, absent in train): {n_pos_cands}")

    if G.number_of_nodes() == 0:
        print("\n✗ Graph is empty — no papers matched the concept filter.")
        print("  Increase max_pages or lower the concept score threshold.")
        return

    # 3. Sample
    print(f"\n⚙  Sampling {CFG['n_pos']} pos + {CFG['n_neg']} neg pairs…")
    dataset = sample_pairs(G, test_co, test_cites,
                           CFG["n_pos"], CFG["n_neg"], CFG["seed"])
    n_p = sum(1 for d in dataset if d["label"] == 1)
    n_n = sum(1 for d in dataset if d["label"] == 0)
    print(f"✓ Dataset         : {n_p} positives, {n_n} negatives")

    if n_p == 0:
        print("\n✗ Zero positive pairs sampled.")
        print("  Possible causes: very small corpus, strict concept filter,")
        print("  or cutoff_T has no absent-then-appeared pairs.")
        print("  Try: increase max_pages to 15-20 or lower cutoff_T to 2016.")
        return

    # 4. Score
    print()
    scored = score_all(dataset, G, CFG["max_path_len"], CFG["min_aa"])
    plaus_n = sum(1 for d in scored if d["plausible"])
    print(f"✓ Pairs passing gate: {plaus_n}/{len(scored)} "
          f"({100*plaus_n/len(scored):.1f}%)")

    # 5. Results
    rq1 = show_rq1(scored)
    rq2 = show_rq2(scored)
    rq3 = show_rq3(scored)

    # 6. Verdict
    banner("PILOT VERDICT")
    passes = 0
    nova_au = rq1.get("NovaGap-Lite (Gated)", {}).get("auroc", 0)
    best_bl = max(
        {k: v for k, v in rq1.items() if "Gated" not in k}.items(),
        key=lambda x: x[1]["auroc"]
    )
    delta = nova_au - best_bl[1]["auroc"]

    if delta > 0:
        print(f"  ✓ RQ1 PASSED  — NovaGap-Lite AUROC {nova_au:.4f} beats best "
              f"baseline ({best_bl[0]}: {best_bl[1]['auroc']:.4f}, Δ={delta:+.4f})")
        passes += 1
    else:
        print(f"  ✗ RQ1 FAILED  — delta {delta:+.4f} vs best baseline.")

    if rq2.get("rho", 0) > 0:
        print(f"  ✓ RQ2 PASSED  — Spearman ρ = {rq2['rho']:+.4f} (positive)")
        passes += 1
    else:
        print(f"  ✗ RQ2 FAILED  — ρ = {rq2.get('rho',0):+.4f}")

    if rq3.get("prec_gate", 0) > rq3.get("prec_raw", 0):
        print(f"  ✓ RQ3 PASSED  — gate precision {rq3['prec_gate']:.4f} > "
              f"ungated {rq3['prec_raw']:.4f}")
        passes += 1
    else:
        print(f"  ✗ RQ3 FAILED  — gate did not improve precision.")

    print(f"\n  Score: {passes}/3 research questions supported.")
    if passes == 3:
        print("  ★ STRONG VERIFICATION — All 3 RQs hold. Full experiment justified.")
    elif passes == 2:
        print("  ✓ GOOD SIGNAL — 2/3 RQs hold. Proceed; address the weak RQ in paper.")
    elif passes == 1:
        print("  ⚠ WEAK — Only 1 RQ holds. Increase corpus size (max_pages→20).")
    else:
        print("  ✗ NO SIGNAL — Check data quality and concept normalization.")

    print("\n  Tip: re-run with max_pages=20 for the full experiment (all 3 cutoffs).")


if __name__ == "__main__":
    main()
