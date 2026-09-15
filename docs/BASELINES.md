# Baseline and evaluation contract

All implementations operate on the same frozen Boolean query oracle. The hidden edges and full response table are never supplied to the scheduling algorithm. The ground-truth negative-cover optimizer is strictly evaluation-only.

## Shared machinery

All methods share the declared component universe, rank ceiling, intervention-size ceiling, query cache, negative-subset inference, positive-superset inference, and residual candidate ledger. They avoid pools containing a known minimal failure set and avoid pools with zero residual coverage. This makes the random controls stronger than unconditioned random-mask sampling.

Every reported query count is the number of distinct nonempty masks actually labeled. A harmless empty mask is known by definition. Counts include positive-pool shrinking and validation calls; caching cannot hide a first oracle evaluation. Timing starts after AuditSpace construction and therefore must not be interpreted as end-to-end algorithm time. Runtime comparisons are not a paper claim.

## Methods

- **Coverage-directed (greedy):** exact enumeration of allowed informative masks, maximum residual candidate coverage, random seed-dependent tie order, followed by randomized deletion shrinking. This is the method in the main query theorem.
- **Random + deletion:** uniform choice among the largest informative masks avoiding found edges, followed by the same deletion extractor. The principal paired comparison isolates the scheduler rather than changing both scheduler and extractor.
- **SIGHT-style:** the same largest-informative random scheduler, with binary-prefix extraction followed by deletion verification. This is a monotone, full-recovery adaptation, not an exact implementation of every detail of the published one-shot SIGHT protocol. The extra verification can be expensive when edges are large.
- **RC-style:** random reductions toward the declared rank, twenty tries per reduction stage, then increasing-cardinality search within a reduced positive pool. A failed extraction is retained and retried, not silently dropped; after three failures on a pool, deletion is used. The outer common ledger differs from the original Random Chemistry sampling protocol.
- **Increasing order:** asks the smallest still-informative masks of order at most r; benefits from the same inference and caching as other methods. It is not artificially required to query all masks.
- **Pairs only:** restricts interventions to size at most two; never treats unexplored rank-three candidates as excluded.

SIGHT and Random Chemistry references: Clarfeld and Eppstein, *Group-Testing on Hypergraphs with Variable-Cost Tests: A Power Systems Case Study*, arXiv:1909.04513. The paper labels the implementations as adaptations. Results are not claimed to establish dominance over optimized native implementations of those methods.

## Comparisons not performed

We did not run the published transformer CoAx pipeline, SMobius, an external pretrained-transformer benchmark, or a white-box formal verifier. They solve related but differently specified tasks. No result in this release is advertised as a head-to-head improvement over those systems.

## Neural validation

Each small model is trained on all 256 inputs of a finite Boolean domain. The clean accuracy measurement is in-domain fit, not held-out generalization. There are eight intervention groups, four ReLU units per group. Every mask/input combination is evaluated to establish experimental truth. The audit receives values only when it queries a mask. Ground-truth enumeration cost is separate from revealed query cost and is reported explicitly.

Positive readout with zero replacement is structurally monotone; signed readout and mean replacement are not guaranteed monotone. All forty tested conditions happened to be monotone under the measured threshold. This finite observation is not a universal assumption justification. Separate cancellation and rank-misspecification controls deliberately violate assumptions.

## Stopping and ground truth

`queries_first_full` / `first_full` is an evaluator-only diagnostic computed after the experiment: the audit does not know when it has found all true edges. Its actual stopping rule is residual-candidate exhaustion and conditional certificate verification. Precision, recall, prediction agreement, and completeness are different quantities and are not substituted for one another.

Reported synthetic uncertainty is standard error over twenty seed-wise averages across the fixed six-family benchmark, not an inference over all possible circuit families. The observed aggregate query reduction is descriptive. Neural comparisons use ten seeds per architecture/patch condition and do not show a uniform greedy advantage.
