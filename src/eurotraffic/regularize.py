"""Per-hour flow-consistency regularization of street vehicle density.

The model predicts each street independently, so the per-hour counts at an
intersection can be physically inconsistent. Every vehicle on an edge must enter or
leave via another edge at each endpoint, so at any vertex no single edge may carry
more flow than the sum of the others ("no dominating edge" — a relaxed Kirchhoff
condition). We enforce this softly, **per hour**, over the whole road graph, by
optimizing the full per-hour density field.

Variables: ``C[i, h] ≥ 0`` = vehicles on edge ``i`` at hour ``h`` (``n × 24``),
warm-started from the prior ``C0[i,h] = a_i · g_{κ_i,h}`` (daily volume × class diurnal
curve — i.e. B1 applied first). With city scale ``σ = median(a₀⁺)`` minimize

    F(C) = Σ_{i,h} w_i ((C−C0)/σ)²                                  (DATA, capacity-aware)
         + μ_flow · Σ_h Σ_{v:deg≥2} Σ_{e∈E_v} [max(0, 2C[e,h] − S_v(h))]² /σ²  (FLOW)
         + μ_time · Σ_i Σ_h ((r[i,h] − r[i,h−1])/σ)²,  r = C − C0       (TIME, cyclic)

where ``S_v(h) = Σ_{e∈E_v} C[e,h]``.
* DATA keeps C near the prior; weight ``w_i`` is large for sensor-anchored streets
  (pinned) and ``(cap_med/cap_i)²`` otherwise, with ``cap_i = lanes·maxspeed`` — so
  high-capacity roads adjust cheaply and small roads stay put.
* FLOW forbids a dominating edge per vertex & hour (μ_flow = tunable strictness).
* TIME keeps the *correction* smooth over the day (no spiky hour-to-hour changes
  without evidence) while preserving the legitimate diurnal shape from C0.

Convex with ``C ≥ 0`` → L-BFGS-B with an analytic, vectorized gradient.

Deferred extension hook: an air-pollution term ``ν·Σ_k (Σ_i K(k,i)C[i,h] − p_k(h))²``
(diffusion kernel ``K`` vs measured pollution ``p``) could be added via ``extra_term``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .class_curves import curve_for

N_HOURS = 24


def _endpoints(wkt: str, precision: int = 6) -> tuple[tuple, tuple]:
    """First and last coordinate of a ``LINESTRING (...)`` WKT, rounded to a node key."""
    inner = wkt[wkt.index("(") + 1 : wkt.rindex(")")]
    pts = inner.split(",")
    fx, fy = pts[0].split()[:2]
    lx, ly = pts[-1].split()[:2]
    a = (round(float(fx), precision), round(float(fy), precision))
    b = (round(float(lx), precision), round(float(ly), precision))
    return a, b


def build_incidence(geoms: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Vertex→edge incidences for vertices of degree ≥ 2 (all classes).

    Returns ``(inc_vertex, inc_edge, n_vertices)``: parallel arrays, one entry per
    (vertex, incident edge) pair. An edge appears once per qualifying endpoint.
    """
    node_edges: dict[tuple, list[int]] = defaultdict(list)
    for i, g in enumerate(geoms):
        if not g:
            continue
        try:
            a, b = _endpoints(g)
        except (ValueError, IndexError):
            continue
        node_edges[a].append(i)
        if b != a:
            node_edges[b].append(i)

    inc_v: list[int] = []
    inc_e: list[int] = []
    vid = 0
    for edges in node_edges.values():
        if len(edges) < 2:  # dead-ends can't balance flow
            continue
        for e in edges:
            inc_v.append(vid)
            inc_e.append(e)
        vid += 1
    return (np.asarray(inc_v, dtype=np.int64),
            np.asarray(inc_e, dtype=np.int64), vid)


def _curve_matrix(osm_types: list, curves: dict) -> np.ndarray:
    """Per-edge 24-h diurnal weight matrix ``G[n, 24]`` (each row sums to 1)."""
    return np.array([curve_for(t, curves) for t in osm_types], dtype=float)


def _data_weights(capacity: np.ndarray, measured: np.ndarray, measured_weight: float) -> np.ndarray:
    """Capacity-aware fidelity stiffness: small for big roads, large (pinned) for sensors."""
    cap = np.asarray(capacity, dtype=float)
    valid = cap[cap > 0]
    cap_med = float(np.median(valid)) if valid.size else 1.0
    cap = np.where(cap > 0, cap, cap_med)
    w = np.clip((cap_med / cap) ** 2, 0.1, 10.0)  # predicted stiffness ∝ 1/capacity²
    return np.where(measured, measured_weight, w)


def regularize_hourly(
    aadt: np.ndarray,
    geoms: list[str],
    osm_types: list,
    curves: dict,
    capacity: np.ndarray,
    measured: np.ndarray,
    mu_flow: float = 0.2,
    mu_time: float = 0.5,
    measured_weight: float = 50.0,
    maxiter: int = 150,
    extra_term: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None,
) -> np.ndarray:
    """Return the regularized per-hour density field ``C[n, 24]``."""
    a0 = np.maximum(np.asarray(aadt, dtype=float), 0.0)
    n = a0.size
    G = _curve_matrix(osm_types, curves)        # n x 24, rows sum to 1
    C0 = a0[:, None] * G                          # prior per-hour field

    inc_v, inc_e, n_v = build_incidence(geoms)
    if inc_e.size == 0:
        return C0

    w = _data_weights(capacity, measured, measured_weight)[:, None]   # n x 1
    pos = a0[a0 > 0]
    sigma2 = float(np.median(pos)) ** 2 if pos.size else 1.0

    def obj_grad(x: np.ndarray):
        C = x.reshape(n, N_HOURS)
        # DATA
        diff = C - C0
        f = float(np.sum(w * diff * diff) / sigma2)
        grad = (2.0 * w * diff) / sigma2

        # FLOW (per hour, on C directly)
        for h in range(N_HOURS):
            ce = C[inc_e, h]
            Sv = np.bincount(inc_v, weights=ce, minlength=n_v)
            hinge = np.maximum(2.0 * ce - Sv[inc_v], 0.0)
            if not hinge.any():
                continue
            f += float(hinge @ hinge) * (mu_flow / sigma2)
            Hv = np.bincount(inc_v, weights=hinge, minlength=n_v)
            contrib = 4.0 * hinge - 2.0 * Hv[inc_v]
            grad[:, h] += (mu_flow / sigma2) * np.bincount(inc_e, weights=contrib, minlength=n)

        # TIME (cyclic roughness of the correction r = C − C0)
        r = C - C0
        dr = r - np.roll(r, 1, axis=1)
        f += float(np.sum(dr * dr)) * (mu_time / sigma2)
        grad += (mu_time / sigma2) * 2.0 * (2.0 * r - np.roll(r, 1, axis=1) - np.roll(r, -1, axis=1))

        if extra_term is not None:
            ef, eg = extra_term(C)
            f += ef
            grad = grad + eg
        return f, grad.ravel()

    res = minimize(
        obj_grad, C0.ravel(), jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * (n * N_HOURS), options={"maxiter": maxiter},
    )
    return np.maximum(res.x.reshape(n, N_HOURS), 0.0)
