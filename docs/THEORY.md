# Theory ledger — proved statements and their scope

Date: 2026-09-15. These are self-contained derivations, not independently reviewed
proofs or claims of historical priority. The negative-cover principle is classical:
Angluin and Chen, *Learning a Hidden Hypergraph*, JMLR 7 (2006), Section 3.
Greedy harmonic charging is also classical. The intended contribution is their
certificate-relative, cross-discovery-phase analysis under an intervention ceiling,
plus a precise account of rank-dependent claims in mechanistic auditing.

## 1. Definitions

Let H be a nonempty-edge antichain on [n], possibly the empty family. Its oracle is
B(A)=1 iff some e in H is contained in A. Let r be the **declared** maximum edge
size; all alternative hypotheses are monotone functions with minimal positive
sets of size at most r. There is no supplied edge-count bound. Queries satisfy
|A|<=b. We assume B(empty)=0, exact labels, and a fixed intervention policy and
behavioral threshold. Results concern intervention-oracle access, not arbitrary
inspection of neural weights.

Let V_r={T:1<=|T|<=r}, I_r(H)={T in V_r:B(T)=0}, M=|I_r(H)|, s=|H|.
Let C=C_{b,r}(H) be the minimum number of true-negative queries of size <=b whose
subsets cover I_r(H). C=0 when I_r is empty, and C=infinity if no cover exists.
Let alpha(H) be the largest size of a negative mask and rho(H)=max|e| (rho=0
for the empty hypergraph). All quantities in these definitions are properties
of the true oracle; the discovery algorithm is not given them.

## 2. Sharp identifiability radius

**Theorem.** H is uniquely determined among all rank-at-most-r monotone oracles
by its responses to all masks of size <=b iff

    b >= max(rho(H), min(r, alpha(H))).

**Proof.** If some e has size >b, deleting e from H leaves every allowed response
unchanged. If a negative mask has size at least b+1 and r>=b+1, select a negative
T of size b+1 and add the term 1[T subset A]. This changes the function at T,
preserves all responses of size <=b, and has a rank-at-most-r minimal antichain.
Conversely, under the stated inequality every true edge is queryable and every
negative T in V_r has size <=b. Querying those sets classifies every T in V_r;
a rank-at-most-r monotone function is determined by these labels. QED.

Thus b<r prevents a uniform guarantee over the hypothesis class, but does not
prevent identification of every particular member of that class. An example is
six disjoint failure pairs: rho=2, alpha=6, so rank-unrestricted certification
requires and can use b=6, even though r=n=12.

## 3. Exact certificate complexity

**Theorem.** Whenever the radius condition holds, the smallest truthful transcript
uniquely identifying H has exactly C+s queries. Otherwise no finite identifying
transcript exists in the allowed query domain.

**Lower bound.** Every negative candidate T in I_r must be contained in a queried
negative mask. Otherwise adding T as an extra positive term changes the oracle
but preserves the whole transcript; minimization of the resulting DNF maintains
rank <=r. Hence at least C negative queries are necessary. For each e in H,
some positive query must contain e and no other true edge, or the transcript
cannot distinguish H from H without e. One positive query can isolate at most
one edge. Hence at least s positive queries are necessary.

**Upper bound.** Query the s true minimal edges and an optimal negative cover.
For any alternative rank-at-most-r monotone function, every true edge remains
positive. Each negative T in V_r remains negative because it lies inside an
observed negative query. Hence the two functions agree on V_r and everywhere.
This is an oracle-dependent *certificate*, not an algorithm that already knows
which optimal queries to ask. QED.

## 4. Residual certificate checker

For verified found edges K and negative masks N, maintain

    U(K,N)={T in V_r: no e in K is contained in T,
                         and T is contained in no mask in N}.

An edge is verified by a positive witness and negative witnesses for each
single-element deletion. Assume every queried positive contains an edge in K.
Then U empty is sufficient for full identification in the declared class.
If U is nonempty, B_K and B_K OR 1[T subset A] for a residual T agree on all
observations but disagree at T. Minimality witnesses prevent a residual T from
being a strict subset of a found edge. This is a conditional certificate: its
validity does not prove global monotonicity, the rank bound, or correct labels.

## 5. Coverage-directed audit and global harmonic bound

At each outer iteration choose a mask avoiding every found edge and maximizing
how many residual candidates it contains. A negative answer removes its subsets.
A positive answer is shrunk by single-component restorations to a minimal edge,
recording all negative queries along the way. Then remove every candidate
containing that edge. Stop only when U is empty.

**Theorem.** Assume the radius condition, exact labels and exact maximization.
Let d=min(M, sum_{k=1}^{min(r,b)} binom(b,k)), and h_d=sum_{j=1}^d 1/j, h_0=0.
Then the algorithm is correct and uses at most

    Q <= h_d C + (b+1)s

oracle calls; caching can only reduce the number of distinct calls.
Consequently Q/(C+s) <= max(h_d,b+1). No polynomial-time maximizer is claimed.
The enumeration implementation is designed for small intervention spaces.

**Proof of correctness and progress.** All registered sets are new true edges,
since their parent pools avoid previously found edges. Deletion is minimal under
monotonicity: if removing i tested negative earlier, any later subset with i
removed is also negative. If a residual candidate is positive, it contains an
undiscovered queryable edge; that edge supplies a feasible positive-score query.
If a residual candidate is negative, an optimal negative-cover member containing
it supplies a feasible positive-score query. Thus each iteration progresses and
the process ends with an empty residual. All original positive pools are explained
by extracted edges. The transcript checker then proves completeness.

**Proof of query bound.** Fix an optimal true-negative cover F of size C. Assign
every T in I_r to one cover mask containing T; this partitions I_r into blocks
P_A with |P_A|<=d. True-negative candidates disappear only through negative
queries, not through finding true positive edges. Every mask in F remains
feasible throughout all edge-discovery phases. On an outer negative iteration,
let k be the number of residual candidates in the chosen mask. Every one of
these k candidates is a true negative. Greedy optimality implies k is at least
the number q of unresolved elements in any one block P_A. Charge 1/k to each
newly covered candidate. If t elements of a block with q remaining elements
are covered on that iteration, its charge is at most t/q, which is at most
sum_{j=q-t+1}^q 1/j. Negative queries during edge extraction can remove elements
at zero charge. Therefore each block accumulates at most h_|P_A|<=h_d, across
**all** discovery phases together. The number of outer negative calls is at
most C h_d. There are at most s outer positive calls, and each needs at most b
additional deletion calls. Adding them proves the result. QED.

If the optimizer achieves at least 1/a of the best coverage score for a>=1,
the same proof gives a h_d C+(b+1)s. Merely running a heuristic does not certify
its approximation factor; this corollary does not assign one to an unproved
optimizer.

## 6. Rank uncertainty can be exponentially expensive

Let H consist of m>=2 disjoint pairs on n=2m vertices. All vertices participate
and the actual rank is exactly two. For b>=m, maximal negative masks select
exactly one vertex from each pair, so they correspond to binary strings of
length m. For declared rank r, define t=min(r,m).

**Theorem.** C_{b,r}(H) is the minimum size of a binary strength-t covering array
with m columns: every assignment on any t distinct columns must occur in a row.
In particular,

    C_{b,n}(H)=2^m,
    C_{b,2}(H)<=2+2 ceil(log2 m),
    C_{b,r}(H)>=2^t.

For r=2 the m positive pair witnesses plus the logarithmic negative cover form
a complete certificate. Without a trusted low-rank bound, the same target
requires 2^m negative queries. The covering-array object itself is classical;
we use it to quantify the dependence of an audit claim on its rank assumption.

**Proof.** A negative candidate contains at most one vertex from any pair.
Extending any negative query to a maximal negative mask does not hurt coverage.
A row covers precisely the partial assignments it extends; thus independent
cover and covering-array conditions are identical. For t=m every binary string
needs its own row, giving equality 2^m. For t=2 label the m columns by distinct
ceil(log2 m)-bit words. Use the all-zero and all-one rows, plus each bit-position
row and its complement. Equal assignments occur in the constant rows; distinct
columns differ at some bit, giving both unequal assignments. For any fixed
collection of t columns, its 2^t assignments require at least 2^t rows. QED.

For b<m and 1<=t=min(r,m)<=b, an additional counting bound is

    C >= ceil(2^t binom(m,t) / binom(b,t)).

One allowed negative query covers at most binom(b,t) oriented t-subsets.
If b<t then C is infinite. This is a bound on negative certification; positive
edges must also be queryable.

## 7. Single hidden group: elementary locality limit

For B_T(A)=1[T subset A] with |T|=k, b<k makes the hidden edge indistinguishable
from the empty hypergraph. If b>=k, an all-negative transcript certifying absence
of any such edge needs at least ceil(binom(n,k)/binom(b,k)) queries. A size-b
query covers at most binom(b,k) possible targets. Adaptive choices do not change
this argument on the all-negative branch.

Uniform independent size-b pools detect a fixed T with probability
p=binom(b,k)/binom(n,k) per query; Q-query miss probability is (1-p)^Q. This is
an elementary covering calculation, not a novelty claim.

## 8. Noise-safe lifting (standard concentration, not a new statistical method)

For the i-th distinct queried mask draw fresh independent loss observations in
[0,1]. At sample count t use radius

    epsilon(i,t)=sqrt(log(pi^4 i^2 t^2/(18 delta))/(2t)).

Classify harmful if mean-epsilon>tau, harmless if mean+epsilon<=tau, and otherwise
continue sampling or abstain at the sample cap. Conditional on the adaptive past,
Hoeffding gives failure at most 36 delta/(pi^4 i^2 t^2). Summing over all positive
integers i,t gives delta. On the resulting simultaneous event all labels are
correct and all deterministic theory applies. This requires fresh sampling or a
separately justified reuse scheme; arbitrary adaptive reuse of one fixed sample
is not justified. A margin gamma from tau permits termination once
2 epsilon(i,t)<gamma. Distinct masks and total mask-input evaluations are separate
resources. Failure to resolve a label must not be treated as a negative answer.

## 9. A neural setting satisfying the assumption exactly

For a binary network with grouped nonnegative hidden activations and a nonnegative
final linear readout, zero-ablation removes nonnegative logit contributions.
For each input, predictions along nested ablations can change only from positive
to negative. Disagreement with the unablated prediction is therefore monotone.
Averaging disagreement and thresholding preserves monotonicity. Signed readouts
or mean replacement do not satisfy this argument in general. No claim is made
about unrestricted transformers, semantic equivalence, on-manifold edits, or task
accuracy under a different distribution.
