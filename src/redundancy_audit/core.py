"""Intervention-query algorithms. No routine has access to the true hidden edges.

A mask bit set to one denotes an ablated component. Oracle values must be Boolean.
Certificates are CONDITIONAL on monotonicity and the declared rank upper bound.
The full candidate/query enumeration is intended for small controlled experiments.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import random
import time
from typing import Callable, Iterable


def mask_of(items: Iterable[int]) -> int:
    out = 0
    for i in items:
        i = int(i)
        if i < 0:
            raise ValueError('Negative component index')
        out |= 1 << i
    return out


def bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if mask & (1 << i)]


def masks_upto(n: int, r: int) -> list[int]:
    return [mask_of(c) for k in range(1, min(n, r) + 1)
            for c in combinations(range(n), k)]


def hypergraph_oracle(edges: Iterable[int]) -> Callable[[int], bool]:
    es = tuple(edges)
    if any(e <= 0 for e in es):
        raise ValueError('Edges must be nonempty masks')
    return lambda a: any((e & a) == e for e in es)


def minimal_edges(table: Iterable[bool], n: int) -> list[int]:
    """Global inclusion-minimal positives; valid also for non-monotone tables."""
    a = list(table)
    if len(a) != 1 << n or a[0]:
        raise ValueError('Expected 2**n entries and a harmless empty mask')
    es: list[int] = []
    for m in sorted(range(1, 1 << n), key=lambda m: (m.bit_count(), m)):
        if a[m] and not any((e & m) == e for e in es):
            es.append(m)
    return es


def monotonicity_violations(table: Iterable[bool], n: int) -> tuple[int, int]:
    a = list(table)
    bad = sum(bool(a[m]) and not bool(a[m | (1 << i)])
              for m in range(1 << n) for i in range(n) if not m & (1 << i))
    return int(bad), n * (1 << (n - 1))


@dataclass
class AuditSpace:
    n: int
    r: int
    b: int
    candidates: list[int] = field(init=False)
    queries: list[int] = field(init=False)
    covers: dict[int, int] = field(init=False)
    supersets: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not (1 <= self.r <= self.n and 1 <= self.b <= self.n):
            raise ValueError('Require 1 <= r,b <= n')
        self.candidates = masks_upto(self.n, self.r)
        self.queries = masks_upto(self.n, self.b)
        idx = {a: i for i, a in enumerate(self.candidates)}
        self.covers = {0: 0}
        for a in self.queries:
            cover = 0
            aa = bits(a)
            for k in range(1, min(self.r, len(aa)) + 1):
                for c in combinations(aa, k):
                    cover |= 1 << idx[mask_of(c)]
            self.covers[a] = cover

    def upward_candidates(self, e: int) -> int:
        if e not in self.supersets:
            self.supersets[e] = sum(1 << i for i, t in enumerate(self.candidates)
                                   if t & e == e)
        return self.supersets[e]


@dataclass
class AuditResult:
    edges: list[int]
    query_count: int
    cache_hits: int
    outer_negative: int
    outer_positive: int
    unresolved: int
    status: str
    elapsed_seconds: float
    transcript: list[tuple[int, bool]]
    trace: list[dict]
    rank_violation: bool = False

    def summary(self) -> dict:
        return {k: v for k, v in vars(self).items() if k not in ('trace', 'transcript')}


class QueryLimit(Exception):
    pass


class AmbiguousQuery(Exception):
    """A noisy oracle abstained; its answer must not enter the certificate."""
    pass


class Ledger:
    def __init__(self, space: AuditSpace, oracle: Callable[[int], bool], limit: int):
        self.space, self.oracle, self.limit = space, oracle, limit
        self.unresolved = (1 << len(space.candidates)) - 1
        self.edges: list[int] = []
        self.answers: dict[int, bool] = {0: False}
        self.transcript: list[tuple[int, bool]] = []
        self.trace: list[dict] = []
        self.cache_hits = self.outer_negative = self.outer_positive = 0
        self.rank_violation = False

    def query(self, a: int) -> bool:
        if a < 0 or a >= (1 << self.space.n) or a.bit_count() > self.space.b:
            raise ValueError('Query violates declared intervention domain')
        if a in self.answers:
            self.cache_hits += 1
            return self.answers[a]
        if len(self.transcript) >= self.limit:
            raise QueryLimit
        y = self.oracle(a)
        if y is None:
            raise AmbiguousQuery
        y = bool(y)
        self.answers[a] = y
        self.transcript.append((a, y))
        if not y:
            self.unresolved &= ~self.space.covers[a]
        return y

    def register(self, e: int) -> None:
        if e == 0 or e in self.edges:
            raise ValueError('Extractor returned empty or duplicate edge')
        self.edges.append(e)
        self.unresolved &= ~self.space.upward_candidates(e)
        self.rank_violation |= e.bit_count() > self.space.r

    def snapshot(self) -> None:
        self.trace.append(dict(queries=len(self.transcript), edges=self.edges.copy(),
                               unresolved=self.unresolved.bit_count()))

    def positives_explained(self) -> bool:
        return all(not y or any(e & a == e for e in self.edges)
                   for a, y in self.transcript)


def shrink_deletion(a: int, ledger: Ledger, rng: random.Random) -> int:
    order = bits(a)
    rng.shuffle(order)
    for i in order:
        candidate = a & ~(1 << i)
        if ledger.query(candidate):
            a = candidate
    return a


def shrink_sight(a: int, ledger: Ledger, rng: random.Random) -> int:
    """SIGHT-inspired binary-prefix extraction with deletion verification.

    This monotone specialization also verifies minimality by deletion. The outer
    full-recovery scheduler is shared across baselines, not the original SIGHT
    one-shot random initialization. See docs/BASELINES.md.
    """
    remaining = bits(a)
    rng.shuffle(remaining)
    d = 0
    while remaining:
        lo, hi = 0, len(remaining) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if ledger.query(d | mask_of(remaining[:mid + 1])):
                hi = mid
            else:
                lo = mid + 1
        d |= 1 << remaining[lo]
        if ledger.query(d):
            return shrink_deletion(d, ledger, rng)
        remaining = remaining[:lo]
    raise RuntimeError('SIGHT invariant failed: oracle may be non-monotone')


def shrink_rc(a: int, ledger: Ledger, rng: random.Random,
              tries: int = 20) -> int | None:
    """Random Chemistry reductions, then increasing-size bottom-up search."""
    r = ledger.space.r
    while a.bit_count() > r:
        target = max(r, (a.bit_count() + 1) // 2)
        aa = bits(a)
        for _ in range(tries):
            c = mask_of(rng.sample(aa, target))
            if ledger.query(c):
                a = c
                break
        else:
            return None
    aa = bits(a)
    for k in range(1, len(aa) + 1):
        subs = list(combinations(aa, k))
        rng.shuffle(subs)
        for sub in subs:
            m = mask_of(sub)
            if ledger.query(m):
                return m
    return None


def audit(space: AuditSpace, oracle: Callable[[int], bool], *,
          method: str = 'greedy', seed: int = 0, max_queries: int = 10000) -> AuditResult:
    """Run an audit; never use truth labels for stopping or choosing queries.

    Methods: greedy; random (same frontier + deletion); sight; rc;
    exhaustive (increasing order); pairs (only order <=2, no false completeness).
    'complete' means conditional on rank <=r and monotonicity, not unconditional.
    """
    if method not in {'greedy', 'random', 'sight', 'rc', 'exhaustive', 'pairs'}:
        raise ValueError(f'Unknown method {method}')
    if max_queries < 0:
        raise ValueError('Negative query budget')
    t0 = time.perf_counter()
    rng, ledger = random.Random(seed), Ledger(space, oracle, max_queries)
    query_order = space.queries.copy()
    rng.shuffle(query_order)
    status = 'incomplete'
    # Random Chemistry can fail to extract an edge from a positive pool. Such
    # pools are not silently discarded: retain them and eventually use deletion.
    rc_failures: dict[int, int] = {}
    try:
        while ledger.unresolved:
            options = []
            for a in query_order:
                if any(a & e == e for e in ledger.edges):
                    continue
                score = (space.covers[a] & ledger.unresolved).bit_count()
                if score and (method != 'pairs' or a.bit_count() <= 2):
                    if method == 'exhaustive' and a.bit_count() > space.r:
                        continue
                    options.append((a, score))
            if not options:
                status = 'locality_limited' if space.b < space.r else 'incomplete'
                break
            if method == 'greedy':
                a, _ = max(options, key=lambda pair: pair[1])
            elif method in ('exhaustive', 'pairs'):
                size = min(a.bit_count() for a, _ in options)
                a = next(a for a, _ in options if a.bit_count() == size)
            else:
                # Uniform among largest still-informative, known-edge-free pools.
                size = max(a.bit_count() for a, _ in options)
                a = rng.choice([a for a, _ in options if a.bit_count() == size])
            if ledger.query(a):
                ledger.outer_positive += 1
                if method == 'sight':
                    e = shrink_sight(a, ledger, rng)
                elif method == 'rc':
                    e = shrink_rc(a, ledger, rng)
                    if e is None:
                        rc_failures[a] = rc_failures.get(a, 0) + 1
                        if rc_failures[a] < 3:
                            continue
                        e = shrink_deletion(a, ledger, rng)
                else:
                    e = shrink_deletion(a, ledger, rng)
                ledger.register(e)
            else:
                ledger.outer_negative += 1
            ledger.snapshot()
        else:
            status = 'complete'
    except QueryLimit:
        status = 'query_budget'
    except AmbiguousQuery:
        status = 'measurement_ambiguous'
    except RuntimeError:
        status = 'oracle_inconsistency'
    if not ledger.positives_explained() and status == 'complete':
        status = 'unexplained_positive'
    if ledger.rank_violation and status == 'complete':
        status = 'rank_bound_violated'
    return AuditResult(sorted(ledger.edges), len(ledger.transcript), ledger.cache_hits,
                       ledger.outer_negative, ledger.outer_positive,
                       ledger.unresolved.bit_count(), status,
                       time.perf_counter() - t0, ledger.transcript,
                       ledger.trace, ledger.rank_violation)
