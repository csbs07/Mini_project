"""
evaluator.py — NovaGap-Lite
Samples degree-matched positive/negative pairs, scores all baselines,
computes AUROC / Average Precision / Precision@K,
and runs the RQ2 citation association test.
"""

import random
import math
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from metrics import (
    compute_atypicality_zscore,
    compute_plausibility,
    compute_gated_novelty,
)


class TemporalEvaluator:
    def __init__(
        self,
        G_train:        nx.Graph,
        test_cooccur:   Dict[Tuple[str, str], int],
        test_citations: Dict[Tuple[str, str], int],
    ):
        self.G       = G_train
        self.t_co    = test_cooccur
        self.t_cites = test_citations

    # ------------------------------------------------------------------
    # 1.  Build evaluation dataset
    # ------------------------------------------------------------------
    def sample_pairs(
        self,
        n_pos: int = 500,
        n_neg: int = 500,
        seed:  int = 42,
    ) -> List[Dict]:
        random.seed(seed)
        nodes = list(self.G.nodes())

        # Positives: absent in G_train, appear >= 2 times in test window
        positives = []
        for (u, v), cnt in self.t_co.items():
            if cnt >= 2 and self.G.has_node(u) and self.G.has_node(v):
                if not self.G.has_edge(u, v):
                    positives.append((u, v))

        random.shuffle(positives)
        positives = positives[:n_pos]

        # Negatives: absent in G_train AND absent in test window, degree-matched
        negatives  = []
        max_tries  = n_neg * 200

        for attempt in range(max_tries):
            if len(negatives) >= len(positives):
                break
            ref_u, ref_v = positives[attempt % len(positives)]
            deg_u = self.G.degree(ref_u)
            deg_v = self.G.degree(ref_v)

            u = random.choice(nodes)
            v = random.choice(nodes)
            if u == v:
                continue
            if abs(self.G.degree(u) - deg_u) > max(2, int(0.25 * deg_u)):
                continue
            if abs(self.G.degree(v) - deg_v) > max(2, int(0.25 * deg_v)):
                continue

            pair = tuple(sorted([u, v]))
            if not self.G.has_edge(u, v) and self.t_co.get(pair, 0) == 0:
                negatives.append(pair)

        dataset = (
            [{"u": u, "v": v, "label": 1,
              "citations": self.t_cites.get(tuple(sorted([u, v])), 0)}
             for u, v in positives]
            +
            [{"u": u, "v": v, "label": 0, "citations": 0}
             for u, v in negatives]
        )
        random.shuffle(dataset)
        return dataset

    # ------------------------------------------------------------------
    # 2.  Score every pair under every method
    # ------------------------------------------------------------------
    def score_dataset(
        self,
        dataset:      List[Dict],
        max_path_len: int  = 3,
        min_aa:       float = 0.05,
    ) -> List[Dict]:
        scored = []
        for item in dataset:
            u, v = item["u"], item["v"]

            # Baseline 1 — Preferential Attachment
            pa = self.G.degree(u) * self.G.degree(v)

            # Baseline 2 — Adamic-Adar alone
            path_len, aa = compute_plausibility(u, v, self.G)

            # Baseline 3 — Atypicality alone (Uzzi)
            atyp = compute_atypicality_zscore(u, v, self.G)

            # Baseline 4 — "Atomic overlap proxy":
            #   fraction of concepts individually present in G_train
            #   (higher = more covered = LESS novel; we invert for ranking)
            present = sum([1 for c in [u, v] if self.G.has_node(c)])
            atomic_proxy = 1.0 - (present / 2.0)   # 1.0 = both absent = proxy-novel

            # NovaGap-Lite — Gated Novelty
            gated_result = compute_gated_novelty(u, v, self.G, max_path_len, min_aa)

            scored.append({
                **item,
                "s_pref_attach":    pa,
                "s_adamic_adar":    aa,
                "s_atypicality":    atyp,
                "s_atomic_proxy":   atomic_proxy,
                "s_gated_novelty":  gated_result["gated_novelty"],
                "path_length":      path_len,
                "is_plausible":     gated_result["is_plausible"],
            })
        return scored

    # ------------------------------------------------------------------
    # 3.  Evaluation metrics
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(scored: List[Dict]) -> Dict[str, Dict[str, float]]:
        y_true = [d["label"] for d in scored]

        methods = {
            "Preferential Attachment": [d["s_pref_attach"]   for d in scored],
            "Adamic-Adar Alone":       [d["s_adamic_adar"]   for d in scored],
            "Atypicality Alone":       [d["s_atypicality"]   for d in scored],
            "Atomic Overlap Proxy":    [d["s_atomic_proxy"]  for d in scored],
            "NovaGap-Lite (Gated)":    [d["s_gated_novelty"] for d in scored],
        }

        results = {}
        for name, preds in methods.items():
            preds_arr = np.array(preds, dtype=float)
            # Guard: if all predictions identical, AUROC is undefined
            if len(set(preds_arr)) <= 1:
                results[name] = {"AUROC": 0.5, "Avg_Precision": float(np.mean(y_true)), "P@50": 0.0}
                continue

            auroc = roc_auc_score(y_true, preds_arr)
            ap    = average_precision_score(y_true, preds_arr)

            # Precision @ top-50
            top50_idx = np.argsort(preds_arr)[::-1][:50]
            p_at_50   = np.mean([y_true[i] for i in top50_idx])

            results[name] = {
                "AUROC":         round(float(auroc), 4),
                "Avg_Precision": round(float(ap), 4),
                "P@50":          round(float(p_at_50), 4),
            }
        return results

    # ------------------------------------------------------------------
    # 4.  RQ2 — Citation association for realized high-N vs low-N pairs
    # ------------------------------------------------------------------
    @staticmethod
    def rq2_citation_analysis(scored: List[Dict]) -> Dict:
        realized  = [d for d in scored if d["label"] == 1]
        if len(realized) < 4:
            return {"note": "too few realized pairs for RQ2"}

        scores  = np.array([d["s_gated_novelty"] for d in realized])
        cites   = np.array([d["citations"]        for d in realized])

        median_score = float(np.median(scores))
        high_novel   = cites[scores >= median_score]
        low_novel    = cites[scores <  median_score]

        # Spearman rho (manual rank correlation, avoids scipy dependency)
        n   = len(scores)
        rs  = scores.argsort().argsort().astype(float)
        rc  = cites.argsort().argsort().astype(float)
        spearman = float(1 - 6 * np.sum((rs - rc) ** 2) / (n * (n**2 - 1))) if n > 2 else 0.0

        return {
            "n_realized_pairs":         len(realized),
            "median_gated_score":       round(median_score, 3),
            "high_novel_median_cites":  round(float(np.median(high_novel)), 1),
            "low_novel_median_cites":   round(float(np.median(low_novel)),  1),
            "spearman_rho":             round(spearman, 4),
            "interpretation":           (
                "Positive rho supports RQ2: higher combinatorial novelty "
                "correlates with higher citation impact."
                if spearman > 0 else
                "Negative / zero rho: combinatorial novelty does not predict "
                "citation impact in this sample (null result — still publishable)."
            ),
        }

    # ------------------------------------------------------------------
    # 5.  RQ3 — Gate ablation: does the gate reduce mirages?
    # ------------------------------------------------------------------
    @staticmethod
    def rq3_gate_ablation(scored: List[Dict]) -> Dict:
        """
        Among pairs with high atypicality:
          - Without gate: fraction that are positives (realized)
          - With gate applied: fraction that are positives (realized)
        A higher precision WITH the gate supports RQ3.
        """
        threshold = float(np.median([d["s_atypicality"] for d in scored]))

        high_atyp       = [d for d in scored if d["s_atypicality"] >= threshold]
        high_atyp_gated = [d for d in high_atyp if d["is_plausible"]]

        prec_without = float(np.mean([d["label"] for d in high_atyp]))    if high_atyp       else 0.0
        prec_with    = float(np.mean([d["label"] for d in high_atyp_gated])) if high_atyp_gated else 0.0
        mirages_removed = len(high_atyp) - len(high_atyp_gated)

        return {
            "high_atypicality_pairs_total":  len(high_atyp),
            "after_gate_filter":             len(high_atyp_gated),
            "mirages_removed":               mirages_removed,
            "precision_without_gate":        round(prec_without, 4),
            "precision_with_gate":           round(prec_with, 4),
            "gate_improves_precision":       prec_with > prec_without,
            "interpretation": (
                "Gate IMPROVES precision: it reduces mirage candidates. Supports RQ3."
                if prec_with > prec_without else
                "Gate does not improve precision in this sample. "
                "Review max_path_len / min_adamic_adar thresholds."
            ),
        }
