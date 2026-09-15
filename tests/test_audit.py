from itertools import combinations
import pytest
from redundancy_audit.core import (AuditSpace, audit, hypergraph_oracle, mask_of,
                                   minimal_edges, monotonicity_violations)
from redundancy_audit.certificates import check_certificate, certificate_cover
from redundancy_audit.noise import BernoulliConfidenceOracle

@pytest.mark.parametrize('method', ['greedy', 'random', 'sight', 'rc', 'exhaustive'])
@pytest.mark.parametrize('edges', [[], [1], [7, 56], [7, 11, 19], [1, 6, 56]])
def test_exact_recovery(method, edges):
    result = audit(AuditSpace(6, 3, 4), hypergraph_oracle(edges), method=method, seed=7)
    assert result.status == 'complete' and result.edges == sorted(edges)
    assert check_certificate(6, 3, 4, result.edges, result.transcript)['valid']

def test_low_order_abstains():
    result = audit(AuditSpace(6, 3, 2), hypergraph_oracle([7]))
    assert result.status == 'locality_limited'
    assert result.edges == [] and result.unresolved > 0

def test_budget_does_not_claim_completeness():
    result = audit(AuditSpace(8, 3, 4), hypergraph_oracle([7, 56]), max_queries=2)
    assert result.status == 'query_budget' and result.query_count == 2

def test_near_threshold_abstention():
    oracle = BernoulliConfidenceOracle(lambda a: 0.5, max_samples=64)
    result = audit(AuditSpace(4, 2, 3), oracle)
    assert result.status == 'measurement_ambiguous' and result.query_count == 0

def test_rank_misspecification_is_not_a_global_guarantee():
    result = audit(AuditSpace(4, 2, 2), hypergraph_oracle([7]))
    assert result.status == 'complete'  # Only conditional on a FALSE rank bound.
    assert result.edges == []
    result = audit(AuditSpace(4, 2, 4), hypergraph_oracle([7]))
    assert result.rank_violation and result.status == 'rank_bound_violated'

def test_nonmonotone_negative_is_not_valid_downward_evidence():
    table = [False, True, False, False]
    assert monotonicity_violations(table, 2)[0] == 1
    result = audit(AuditSpace(2, 2, 2), lambda a: table[a])
    assert result.status == 'complete'  # Conditional assumptions fail here.
    assert result.edges != minimal_edges(table, 2)

def test_empty_model_cover_is_binomial_when_b_equals_r():
    assert certificate_cover(AuditSpace(5, 2, 2), [])['optimum'] == 10

def test_negative_certificate_necessity():
    result = audit(AuditSpace(5, 2, 2), hypergraph_oracle([]))
    for j in range(len(result.transcript)):
        tr = result.transcript[:j] + result.transcript[j+1:]
        assert not check_certificate(5, 2, 2, [], tr)['valid']

def test_invalid_parameters():
    with pytest.raises(ValueError): AuditSpace(3, 0, 2)
    with pytest.raises(ValueError):
        audit(AuditSpace(3, 2, 2), lambda a: False, method='unknown')

@pytest.mark.parametrize('pairs', [2, 3, 4, 5, 6])
def test_pair_rank_two_explicit_logarithmic_certificate(pairs):
    from math import ceil, log2
    from redundancy_audit.core import masks_upto
    n = 2 * pairs
    edges = [3 << (2*j) for j in range(pairs)]
    assignments = [[0]*pairs, [1]*pairs]
    for bit in range(ceil(log2(pairs))):
        row = [(j >> bit) & 1 for j in range(pairs)]
        assignments.extend([row, [1-x for x in row]])
    negatives = sorted(set(sum(1 << (2*j+row[j]) for j in range(pairs))
                           for row in assignments))
    transcript = [(a, False) for a in negatives] + [(e, True) for e in edges]
    certificate = check_certificate(n, 2, pairs, edges, transcript)
    assert certificate['valid']
    assert len(negatives) <= 2 + 2*ceil(log2(pairs))

@pytest.mark.parametrize('pairs', [2, 3, 4])
def test_pair_unrestricted_rank_certificate_is_exponential(pairs):
    n = 2 * pairs
    edges = [3 << (2*j) for j in range(pairs)]
    cover = certificate_cover(AuditSpace(n, n, pairs), edges)
    assert cover['optimum'] == 2**pairs


def test_numpy_indices_become_unbounded_python_bitmasks():
    import numpy as np
    from redundancy_audit.core import mask_of
    mask = mask_of(np.array([1, 70]))
    assert isinstance(mask, int) and mask == 2 + (1 << 70)
