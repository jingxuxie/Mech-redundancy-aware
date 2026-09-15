"""Conservative anytime/any-query confidence wrapper; fresh samples per mask.

This is a standard Hoeffding + summable union-bound construction, not a novel
confidence sequence. Adaptive reuse of the SAME fixed prompt set is unsupported.
"""
from __future__ import annotations
import math
import numpy as np


class BernoulliConfidenceOracle:
    def __init__(self, probability, *, tau=0.5, delta=0.05, seed=0,
                 max_samples=262144, fixed_samples=None):
        if not (0 < tau < 1 and 0 < delta < 1):
            raise ValueError('tau and delta must be in (0,1)')
        self.probability, self.tau, self.delta = probability, tau, delta
        self.rng = np.random.default_rng(seed)
        self.max_samples, self.fixed_samples = max_samples, fixed_samples
        self.total_samples = self.calls = 0
        self.records = []

    def __call__(self, mask):
        self.calls += 1
        i, successes, n = self.calls, 0, 0
        p = float(self.probability(mask))
        if not 0 <= p <= 1:
            raise ValueError('Invalid probability')
        if self.fixed_samples is not None:
            n = self.fixed_samples
            successes = self.rng.binomial(n, p)
            self.total_samples += n
            y = bool(successes / n > self.tau)
            self.records.append((mask, n, y))
            return y
        target = 32
        while n < self.max_samples:
            target = min(target, self.max_samples)
            successes += int(self.rng.binomial(target - n, p))
            self.total_samples += target - n
            n = target
            radius = math.sqrt(math.log(math.pi**4 * i*i * n*n /
                                        (18 * self.delta)) / (2*n))
            mean = successes / n
            if mean - radius > self.tau:
                self.records.append((mask, n, True))
                return True
            if mean + radius <= self.tau:
                self.records.append((mask, n, False))
                return False
            target *= 2
        self.records.append((mask, n, None))
        return None
