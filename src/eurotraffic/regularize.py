"""Graph-based regularization of per-segment AADT.

The model predicts each street independently, so two segments of the same road can
get visibly different volumes. We smooth the estimates over the street-network graph
(segments are adjacent when they share an endpoint / intersection) by minimizing

    sum_i  w_i (x_i - y_i)^2   +   lambda * sum_{(i,j) in E} (x_i - x_j)^2

in log space, where y = log1p(AADT), w_i is a data weight (large for measured-anchored
streets so they barely move), and E is the segment adjacency. The minimizer solves the
sparse SPD system ``(W + lambda L) x = W y`` (L = graph Laplacian), handled with a
conjugate-gradient solve. ``lambda`` trades consistency for fidelity — small
inconsistencies remain by design.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg


def _endpoints(wkt: str, precision: int = 6) -> tuple[tuple, tuple]:
    """First and last coordinate of a ``LINESTRING (...)`` WKT, rounded to a node key."""
    inner = wkt[wkt.index("(") + 1 : wkt.rindex(")")]
    pts = inner.split(",")
    fx, fy = pts[0].split()[:2]
    lx, ly = pts[-1].split()[:2]
    a = (round(float(fx), precision), round(float(fy), precision))
    b = (round(float(lx), precision), round(float(ly), precision))
    return a, b


def segment_adjacency(
    geoms: list[str], groups: list | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the segment graph: two segments are linked if they share an endpoint.

    If ``groups`` is given (e.g. road class per segment), only same-group segments at
    a shared node are linked — so smoothing keeps a road consistent *along itself*
    without averaging a motorway into the residential street it crosses.

    Returns parallel (rows, cols) index arrays (upper-triangular pairs).
    """
    node_segs: dict[tuple, list[int]] = defaultdict(list)
    for i, g in enumerate(geoms):
        if not g:
            continue
        try:
            a, b = _endpoints(g)
        except (ValueError, IndexError):
            continue
        node_segs[a].append(i)
        if b != a:
            node_segs[b].append(i)

    rows: list[int] = []
    cols: list[int] = []
    for segs in node_segs.values():
        if len(segs) < 2:
            continue
        # Partition the segments meeting at this node by group, link within each.
        buckets: dict = defaultdict(list)
        for s in segs:
            buckets[groups[s] if groups is not None else 0].append(s)
        for members in buckets.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    rows.append(members[a])
                    cols.append(members[b])
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def regularize_aadt(
    aadt: np.ndarray,
    weights: np.ndarray,
    geoms: list[str],
    groups: list | None = None,
    lam: float = 1.0,
    rtol: float = 1e-4,
    maxiter: int = 600,
) -> np.ndarray:
    """Return graph-smoothed AADT (same length), anchored by ``weights``.

    ``groups`` (road class per segment) restricts smoothing to same-class neighbours.
    """
    n = len(aadt)
    rows, cols = segment_adjacency(geoms, groups)
    if rows.size == 0:
        return np.asarray(aadt, dtype=float)

    y = np.log1p(np.maximum(np.asarray(aadt, dtype=float), 0.0))
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    M = (sp.diags(weights) + lam * L).tocsr()

    x, info = cg(M, weights * y, x0=y, rtol=rtol, maxiter=maxiter)
    if info != 0:  # not converged → fall back to the unsmoothed estimate
        x = y
    return np.expm1(x)
