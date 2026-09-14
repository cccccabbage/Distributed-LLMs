== FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference <paper-flowspec>

=== Summary

FlowSpec@liuFlowSpecContinuousPipelined2025 is a distributed inference framework for running large
language models across resource-constrained edge devices. It targets autoregressive decoding with
one or a few requests, where pipeline parallelism has too little independent work to keep all stages
busy. FlowSpec combines pipeline parallelism, tree-based speculative decoding, score-based candidate
selection, and continuous verification, pruning, and expansion.

Its central idea is to treat speculative tokens as a continuously replenished stream of work. The
pipeline can verify candidates from one speculative round while the draft device prepares later
candidates, reducing the repeated draining and refilling that otherwise creates pipeline bubbles.
FlowSpec is a scheduling and runtime technique, not a new target-model architecture or draft-model
training method. The experiments use EAGLE and EAGLE-2 as the underlying speculative decoding
frameworks.

=== Issues Addressed

FlowSpec instantiates the constrained-device and model-placement setting described in
@issue-inference-resource-heterogeneity[Distributed-Inference Resource Heterogeneity and Model
  Placement] and the shared pipeline behavior in @issue-pipeline-bubbles-utilization[Pipeline
  Bubbles and Stage Utilization in Distributed Inference]. Its paper-specific challenge is that
existing pipelined speculative decoders can verify by tree depth, replicate and update the full tree
on every stage, or continue processing candidates made stale by an accepted context change.

FlowSpec addresses these specific sources of waste by prioritizing candidates by cumulative
confidence rather than depth, keeping complete tree state on the draft device, removing incompatible
branches early, and continuously generating candidates conditioned on the newest accepted context.
The paper therefore concerns distributed serving efficiency, not federated training or a formal
privacy guarantee.

=== Method

FlowSpec uses an $N+1$ device architecture. Device $D_0$ runs a small learned draft model and owns
the speculative tree. Devices $V_1$ through $V_N$ hold consecutive portions of the large target LLM
and form the verification pipeline:

$ D_0 -> V_1 -> V_2 -> dots -> V_N $

The draft model predicts likely future tokens, while FlowSpec determines how those predictions are
selected, segmented, scheduled, verified, and discarded. This is the distinction from EAGLE: EAGLE
provides the speculative decoding framework and draft model used in the evaluation, whereas FlowSpec
organizes speculative work for a distributed pipeline.

The draft device initially constructs a relatively broad tree of possible future token sequences.
Each node has a confidence $c(n_i)$, and FlowSpec assigns each node a cumulative confidence over its
path:

$ c_"cu"(n_i) = product_(n_j in "path"(n_i)) c(n_j) $

For example, a node reached through probabilities $0.9$ and $0.8$ has path confidence
$0.9 times 0.8 = 0.72$. Candidates are sorted by cumulative confidence, rather than being grouped
strictly by depth. This ordering preserves dependencies because a child has the form
$c_"child" = c_"parent" times p$, where $0 <= p <= 1$. Thus a child cannot outrank its parent, so
the sorted order is also a valid dependency order for the tree.

The ordered candidates are divided into segments $S^(0), S^(1), dots, S^(N)$ and inserted into the
stages of the verification pipeline. Verification is continuous and segmented rather than a single
depth-by-depth tree pass. While stages verify existing segments, $D_0$ can draft later work. This
overlap aims to replace sequential drafting and verification with concurrent execution:

$ "draft generation" parallel "target-model verification" $

After a segment is verified, the target model returns the accepted speculative sequence $S_"acc"$
and a newly sampled token $x_"new"$. FlowSpec checks whether the concatenation $S_"acc" || x_"new"$
remains a path in the current tree. If it does, the same speculative round can continue with useful
candidates already in flight. If it does not, the remaining candidates are inconsistent with the
generated sequence and that round ends.

The draft device then prunes branches that no longer agree with the verified path. Pruning avoids
verification of invalid candidates and removes their associated KV-cache entries. Verification
devices do not maintain full copies of the tree. $D_0$ sends compact token-index information that
identifies which entries should be retained, reducing tree-management communication.

FlowSpec continuously replenishes the tree in two ways. When tokens are accepted, it drafts new
candidates from the latest context, merges them with the surviving tree, and removes duplicate
paths. This context-aware expansion avoids relying only on leaf extensions generated from an old
context, which can make candidates stale. When pruning empties a future segment without producing
new context, FlowSpec expands the existing base tree to additional depths, removes candidates
already present, selects the highest-scoring remaining nodes, and inserts them into future segments.

The resulting loop is verify, accept, prune, expand, and verify again. The purpose is to keep target
model stages supplied with useful speculative microbatches while drafting and verification proceed
in parallel. The design still has an initial fill phase, but it avoids repeatedly draining and
restarting the pipeline during steady-state decoding.

=== Pros and Cons

==== Pros

- The score-based policy can select candidates from different depths while retaining dependency
  order, so promising sequential tokens need not wait for a complete shallow-level pass.
- Segmented continuous verification gives pipeline stages a stream of speculative work and allows
  drafting to overlap with target-model verification. This directly targets bubbles in low-request
  pipeline-parallel inference.
- Draft-device-only tree ownership avoids replicating the complete speculative tree across all
  verification stages. Compact token-index updates also support pruning without full-tree
  synchronization.
- Context-aware expansion refreshes candidates after accepted tokens, while the fallback expansion
  path keeps future segments occupied when pruning changes the tree without changing the context.
- The paper reports approximately $1.45 times$ average speedup for 7B models and $1.70 times$ for
  13B models over Chunk-PP, plus roughly $1.2 times$ to $1.4 times$ improvement over PipeDec. It
  also reports more accepted tokens per speculative decoding round.
- The evaluated pipeline-parallel design is relevant to edge networks where bandwidth is lower than
  that of interconnects such as NVLink or InfiniBand.

==== Cons

- Analysis: speedup remains bounded by draft quality. Frequent rejection reduces the useful work per
  verification round and can turn drafting, cache management, and pipeline scheduling into overhead.
- The $N+1$ architecture assigns a separate device, memory, KV cache, and computation to drafting.
  This cost can reduce the benefit when the draft model is not sufficiently cheap.
- Adding verification stages increases activation-transfer requirements. FlowSpec can overlap some
  communication with computation, but very low bandwidth or high latency remains visible and limits
  scalability.
- The initial pipeline fill still creates a cold start, and the continuous policy introduces runtime
  complexity for confidence scores, segmented scheduling, tree merging and pruning, attention masks,
  position IDs, KV-cache retention, cross-device indices, and asynchronous execution.
- Tree size, initial and expansion depths, candidate count, and segment size require tuning. Larger
  speculation can expose more accepted tokens but also increases drafting and verification work.
- The reported evaluation is limited to five NVIDIA Jetson Orin Nano devices, 7B and 13B models,
  LLaMA2-Chat and Vicuna, and EAGLE-based decoding. The supplied evidence therefore does not
  establish the same behavior for much larger models, datacenter accelerators, other speculative
  decoders, or substantially deeper pipelines.
