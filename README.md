# Finding Is Not Certifying

**Query-Limited Audits of Redundant Mechanisms**  
Research manuscript and reproducible CPU experiments, 15 September 2026.

This project studies recovery of **all minimal harmful ablation sets**, not extraction of one sufficient circuit. It separates finding the actual dependencies from collecting enough evidence to exclude additional dependencies.

## Main results

For a monotone intervention oracle with declared maximum failure order `r`, intervention-size ceiling `b`, true minimal failure family `H`, and no supplied edge-count bound:

1. **Exact identifiability radius:** identification is possible precisely when `b >= max(rank(H), min(r, alpha(H)))`, where `alpha(H)` is the largest harmless mask size.
2. **Exact certificate size:** the smallest truthful identifying transcript contains `C + |H|` queries, where `C` is the minimum bounded-size negative cover of all harmless candidate sets of order at most `r`.
3. **Global adaptive bound:** coverage-directed querying with deletion extraction uses at most `h_d C + (b+1)|H|` queries. Here `h_d` is a harmonic number and `d` bounds the number of harmless candidates covered by one query. The charge spans all discovery phases and permits arbitrary overlap.
4. **Cost of rank uncertainty:** for `m` disjoint failure pairs, an unrestricted-rank negative certificate needs `2**m` masks, whereas a trusted rank-two assumption admits at most `2 + 2*ceil(log2(m))` negative masks, when `b >= m`.

See [the manuscript](paper/main.tex) for full statements and proofs and [the theory note](docs/THEORY.md) for derivations. These statements are **conditional on monotonicity, truthful labels, the specified patching policy, and the declared rank bound**. The algorithm enumerates a finite query space; it is not a polynomial-time large-model discovery method.

The hidden-hypergraph formulation, independent-cover principle, covering arrays, and harmonic set-cover charging are established mathematics. The proposed contribution is the certificate-relative adaptive analysis and its interpretation for bounded mechanistic audits. Historical priority of that precise bound still needs independent review; see [research status](docs/RESEARCH_STATUS.md).

## Completed experiments

| Experiment | Actual scope and result |
|---|---|
| Independent finite theorem verification | 1,848 rank/radius cases over all 167 nonconstant-one monotone functions on four components; 1,254 minimum distinguishing-transcript optimizations. All passed. |
| Synthetic audit comparison | 2,880 audits: six families, four intervention ceilings, twenty seeds, six methods. All five full-recovery methods recovered and certified all 480 cases each. Pair-only auditing did not certify the rank-three class. |
| Matched greedy versus random scheduler | 11.60% reduction in aggregate mean query count; 362 wins, 108 ties, 10 losses across 480 paired cases. Not a universal advantage. |
| Fixed six-pair rank sweep | 480 audits. Optimal negative certificate grows from 6 at declared rank 2 to 64 at rank 6 or 12, although the true model does not change. |
| Radius sweep | 100 audits. All six true pairs were found at every ceiling 2–6, but only ceiling 6 allowed a rank-unrestricted certificate. |
| Trained tiny networks | 20 CPU-trained networks, 40 model/patch conditions, 160 audits; all exact and certified in these fully enumerated conditions. No consistent greedy advantage over neural baselines. |
| Noisy labels | 80 audits. Confidence wrapper: 0/40 false checked certificates; fixed batches of 32: 7/40. Validity comes from the concentration proof, not forty empirical successes. Confidence required substantially more input samples. |
| Assumption controls | Frozen-network cancellation and misspecified-rank examples produce incorrect conditional claims without visible transcript contradictions. |

The GitHub release contains measured summaries, exact figure inputs, generated LaTeX tables, and all reproduction code. The accompanying downloadable research bundle additionally contains the complete raw CSV/JSON records and `results/raw_records.tar.xz`, with checksums in `results/RAW_MANIFEST.json`. Seed-zero network weights and full discrepancy tables are in that raw-record bundle; other models are reproducible from seeds rather than stored checkpoints. Run `make experiments` to regenerate all raw records from a GitHub checkout.

The main successful experiment run took 251.46 seconds in the recorded container; additional rank, radius, and independent-proof checks were separate. This is **not** a hardware-normalized laptop runtime claim. No GPU, LLM API, or pretrained-model download was used.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e '.[experiments]'
make test
make figures
make paper                              # Builds from shipped figure data; needs pdflatex, not BibTeX.
```

To regenerate all experimental records (including from a GitHub-only checkout):

```bash
make experiments
```

`make experiments` includes all synthetic experiments, twenty network fits, rank/radius sweeps, independent finite-hypothesis verification, and assumption controls. The integer optimizer may take longer on other hardware; only solver status zero is reported as an optimum. `requirements-reproduced.txt` records the versions actually used, while `pyproject.toml` lists supported lower bounds.

A minimal use example:

```python
from redundancy_audit import AuditSpace, audit, hypergraph_oracle
from redundancy_audit.certificates import check_certificate

# Two hidden triple-failure sets; the audit sees only the callable oracle.
space = AuditSpace(n=8, r=3, b=4)
oracle = hypergraph_oracle([0b00000111, 0b00111000])
result = audit(space, oracle, method="greedy", seed=0)
certificate = check_certificate(8, 3, 4, result.edges, result.transcript)
print(result.status, result.query_count, certificate["valid"])
```

`complete` is conditional, not a claim that a partial transcript establishes global monotonicity or the rank assumption. Check the returned transcript independently. `locality_limited`, `query_budget`, `measurement_ambiguous`, and rank/consistency diagnostics must not be treated as success.

## Repository map

- `src/redundancy_audit/`: auditable query algorithms, transcript checker, negative-cover optimizer, confidence wrapper.
- `experiments/`: exact synthetic and trained-network experiments, independent theory verifier, table/figure generation.
- `tests/`: 42 passing tests in the recorded run.
- `paper/`: complete research manuscript, bibliography, and generated result tables.
- `docs/BASELINES.md`: baseline adaptations and comparison limits.
- `docs/RESEARCH_STATUS.md`: completed work and remaining submission risks.
- `results/`: measured summaries, figure data, and provenance. The downloadable bundle also includes raw records and their verified archive.

The manuscript is a full first research draft, **not independently reviewed or represented as accepted/submission-ready**. The strongest remaining tasks are an independent proof/priority review, externally sourced circuit validation, and scalable query selection.
