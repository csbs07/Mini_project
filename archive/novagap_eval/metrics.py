"""
metrics.py — NovaGap-Lite
Computes:
  1. Atypicality A(u,v): Uzzi-style z-score (degree-preserving null / PMI fallback)
  2. Plausibility P(u,v): shortest-path length + Adamic-Adar index
  3. Gated Novelty N(u,v): atypicality gated by structural reachability
"""

import math
import networkx as nx
import numpy as np
from typing import Dict, Tuple


def compute_pmi(u: str, v: str, G: nx.Graph, total_papers: int) -> float:
    """PMI(u,v) = log(P(u,v) / (P(u)*P(v))). Returns -5.0 for absent edges."""
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    d_u = G.nodes[u].get("paper_count", G.degree(u))
    d_v = G.nodes[v].get("paper_count", G.degree(v))
    edge_weight = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
    if edge_weight == 0:
        return -5.0
    p_uv = edge_weight / total_papers
    p_u  = d_u / total_papers
    p_v  = d_v / total_papers
    if p_u == 0 or p_v == 0:
        return -5.0
    return float(np.log(p_uv / (p_u * p_v)))


def compute_atypicality_zscore(u: str, v: str, G: nx.Graph) -> float:
    """
    Uzzi-style atypicality.
    z = (actual_weight - expected_weight) / sqrt(expected_weight)
    We INVERT: high positive = lower than expected = ATYPICAL = potentially novel.
    For absent pairs: z is clearly negative -> after inversion -> positive (novel).
    """
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    total_w = G.size(weight="weight")
    if total_w == 0:
        return 0.0
    d_u = G.degree(u, weight="weight")
    d_v = G.degree(v, weight="weight")
    expected  = (d_u * d_v) / (2.0 * total_w)
    actual    = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
    std_approx = math.sqrt(expected) if expected > 0 else 1.0
    z = (actual - expected) / std_approx
    # Invert: high atypicality (rarer than expected) returns high positive
    return float(-z)


def compute_plausibility(u: str, v: str, G: nx.Graph) -> Tuple[int, float]:
    """
    Returns (shortest_path_length, adamic_adar_score).
    path = 999 if no path exists.
    """
    if not G.has_node(u) or not G.has_node(v):
        return (999, 0.0)
    try:
        path_len = nx.shortest_path_length(G, source=u, target=v)
    except nx.NetworkXNoPath:
        path_len = 999

    aa = 0.0
    for w in nx.common_neighbors(G, u, v):
        deg_w = G.degree(w)
        if deg_w > 1:
            aa += 1.0 / math.log(deg_w)

    return (path_len, float(aa))


def compute_gated_novelty(
    u: str,
    v: str,
    G: nx.Graph,
    max_path_len: int = 3,
    min_adamic_adar: float = 0.05,
) -> Dict[str, float]:
    """
    Gated Novelty N(u,v):
      = Atypicality   if pair is structurally reachable (plausible)
      = 0.0           otherwise (mirage / isolated nonsense)
    """
    atyp               = compute_atypicality_zscore(u, v, G)
    path_len, aa_score = compute_plausibility(u, v, G)

    is_plausible = (path_len <= max_path_len) or (aa_score >= min_adamic_adar)
    gated        = atyp if is_plausible else 0.0

    return {
        "concept_a":      u,
        "concept_b":      v,
        "atypicality":    round(atyp, 4),
        "path_length":    path_len,
        "adamic_adar":    round(aa_score, 4),
        "is_plausible":   is_plausible,
        "gated_novelty":  round(gated, 4),
    }
