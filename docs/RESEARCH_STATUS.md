# Research status — 15 September 2026

## Completed in this release

A complete manuscript contains proofs of the instance-wise identifiability radius, exact identifying-transcript complexity, the adaptive global harmonic bound, and the covering-array/rank-uncertainty separation. The confidence-label wrapper and constrained-network monotonicity argument are also proved. All claims are conditional on their stated oracle class, rather than on semantics of a uniquely correct internal algorithm.

Independent exhaustive hypothesis enumeration checked the radius theorem in 1,848 cases. For 1,254 identifiable cases, a separate minimum distinguishing-transcript integer program agrees with the negative-cover certificate formula. An earlier 332-case certificate/recovery check and 42 implementation tests also pass. These tests find counterexamples to implementation or statement errors; they do not replace the general proofs or an independent mathematical review.

The released observations are actual executed results: 2,880 combinatorial audits, 480 fixed-model rank audits, 100 radius audits, 160 audits across 20 CPU-trained neural networks and two patch policies, and 80 noisy audits. Complete raw records, summaries, seed-zero checkpoints, full seed-zero discrepancy tables, and experiment environment metadata are archived in the downloadable research bundle. The GitHub tree contains source, measured summaries and plotting inputs; `make experiments` recreates the full records. No large model, GPU, external dataset, or LLM API was used.

The fixed-pair experiment isolates the central insight: discovery takes about 44 queries across rank declarations, but greedy certification rises from 46.25 to 106 mean queries; the optimum negative certificate rises from 6 to 64. True model complexity does not change.

## What is established prior art

Hidden-hypergraph/monotone-DNF learning, negative independent covers, covering arrays, harmonic set-cover approximation, and adaptive group-testing extraction already exist. Angluin and Chen (JMLR 2006) explicitly use independent covers and iterative discovery. We do not claim these objects or the observation that single ablations can miss redundancy as new.

The candidate contribution is the exact bounded-domain certificate formulation together with a single certificate-relative harmonic charge across all adaptive discovery phases, and the resulting interpretation of rank-sensitive mechanistic completeness. A dedicated historical-priority check of the exact theorem is still needed, particularly against exact-learning and instance-optimal query-learning literature. The paper deliberately avoids a claim of improved general worst-case hidden-hypergraph complexity.

## Submission risks and next work

1. **Independent proof and priority review.** Compare the exact adaptive bound, not just terminology, against existing monotone-DNF algorithms. An external proof audit is desirable, especially for the no-known-edge-count hypothesis class and the radius condition.
2. **Practical baseline scope.** RC/SIGHT are documented full-recovery adaptations, not verified native reproductions. The neural examples are small and fully enumerated. No claim of transformer-level effectiveness is supported.
3. **Computational scaling.** Current query maximization enumerates the candidate/query spaces. Query-efficient is not computationally efficient. A larger-scale method would need an approximate oracle with an explicit coverage guarantee or a justified heuristic.
4. **Assumption relevance.** Partial transcripts cannot establish monotonicity, trustworthy rank bounds, or in-distribution patching. The proposed output must retain its intervention policy and assumptions. Non-monotone extensions are not solved by this manuscript.
5. **Conference preparation.** The current twelve-page research PDF includes references and appendices and is not in a conference-year official style. Author information, external review, and final venue formatting remain to be done.

This is a substantial first research manuscript with real results, not a claim that acceptance, novelty, independent verification, or submission readiness has already been secured.
