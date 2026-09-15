"""Independent certificate checks and small-instance set-cover optimization."""
from __future__ import annotations
from math import comb
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from scipy.sparse import csc_matrix
from .core import AuditSpace, hypergraph_oracle, masks_upto


def residual_candidates(n: int, r: int, edges: list[int],
                        negatives: list[int]) -> list[int]:
    return [t for t in masks_upto(n, r)
            if not any(e & t == e for e in edges)
            and not any(t & a == t for a in negatives)]


def check_certificate(n: int, r: int, b: int, edges: list[int],
                      transcript: list[tuple[int, bool]]) -> dict:
    """Check the certificate using only its transcript, not the hidden model.

    Still assumes global monotonicity and rank<=r. Detecting no inconsistency in
    a partial transcript is not a test establishing either global assumption.
    """
    negative = [a for a, y in transcript if not y] + [0]
    positive = [a for a, y in transcript if y]
    domain = all(0 <= a < 1 << n and a.bit_count() <= b for a, _ in transcript)
    consistent = all(not (p & q == p) for p in positive for q in negative)
    sound = all(0 < e < 1 << n and any(p & e == p for p in positive)
                and all(any((e & ~(1 << i)) & q == (e & ~(1 << i))
                            for q in negative)
                        for i in range(n) if e & (1 << i)) for e in edges)
    explained = all(any(e & p == e for e in edges) for p in positive)
    residual = residual_candidates(n, r, edges, negative)
    rank_ok = all(e.bit_count() <= r for e in edges)
    return dict(valid=bool(domain and consistent and sound and explained and
                           rank_ok and not residual),
                domain=domain, consistent=consistent, minimality_witnesses=sound,
                positives_explained=explained, rank_ok=rank_ok,
                residual_count=len(residual), residual=residual)


def certificate_cover(space: AuditSpace, edges: list[int],
                      time_limit: float = 20.0) -> dict:
    """Solve the full-truth certificate cover for EVALUATION ONLY.

    No audit algorithm calls this routine. LP is a lower bound; MILP output is
    called an optimum only when solver status=0. All candidate orders 1..r count.
    """
    oracle = hypergraph_oracle(edges)
    independent = [t for t in space.candidates if not oracle(t)]
    safe = [a for a in space.queries if not oracle(a)]
    if not independent:
        return dict(optimum=0, lp_bound=0.0, feasible_cover=0, status=0,
                    independent_count=0, greedy_bound=(space.b + 1) * len(edges))
    # Maximal safe masks dominate smaller masks for uniform query costs.
    maximal = [a for a in safe if a.bit_count() == space.b or
               not any(not oracle(a | (1 << i)) for i in range(space.n)
                       if not a & (1 << i))]
    mat = csc_matrix(np.array([[int(t & a == t) for a in maximal]
                              for t in independent], dtype=float))
    if mat.shape[1] == 0 or np.any(np.asarray(mat.sum(axis=1)).ravel() == 0):
        return dict(optimum=None, lp_bound=None, feasible_cover=None, status=2,
                    independent_count=len(independent), greedy_bound=None)
    lp = linprog(np.ones(len(maximal)), A_ub=-mat, b_ub=-np.ones(len(independent)),
                 bounds=(0, None), method='highs')
    sol = milp(np.ones(len(maximal)), integrality=np.ones(len(maximal)),
               bounds=Bounds(0, 1),
               constraints=LinearConstraint(mat, np.ones(len(independent)), np.inf),
               options={'time_limit': time_limit})
    optimum = int(round(sol.fun)) if sol.status == 0 else None
    feasible = int(round(sol.fun)) if sol.fun is not None else None
    cover_width = min(len(independent), sum(comb(space.b, k) for k in range(1, min(space.r, space.b) + 1)))
    harmonic = sum(1 / i for i in range(1, cover_width + 1))
    return dict(optimum=optimum, lp_bound=float(lp.fun) if lp.success else None,
                feasible_cover=feasible, status=int(sol.status),
                independent_count=len(independent),
                greedy_bound=harmonic * optimum + (space.b + 1) * len(edges)
                if optimum is not None else None)
