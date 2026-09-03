== Seesaw: High-throughput LLM Inference via Model Re-sharding <paper-seesaw>

=== Summary

Seesaw@suSeesawHighthroughputLLM2025 is an inference engine for offline, throughput-oriented
workloads. It observes that the two phases described in @background-prefill-decode-kv-cache[Prefill,
  Decode, and the Key-Value Cache] have different bottlenecks and therefore favor different
parallelization strategies. Prefill processes many prompt tokens and can be communication-heavy
under tensor parallelism (TP), whereas decode repeatedly loads model weights for relatively little
computation and benefits from TP. Seesaw therefore uses more pipeline parallelism (PP) for prefill
and more TP for decode, re-sharding the model between phases. The paper reports up to 1.78 times
higher throughput than its evaluated vLLM configuration, with a 1.36 times average improvement.

=== Issues Addressed

Seesaw addresses the mismatch between a static parallelization strategy and the distinct compute and
communication profiles of prefill and decode. TP can incur frequent all-reduce operations over many
prompt-token activations during prefill, while PP can create small decode micro-batches that make
GPUs reload weights repeatedly. A single configuration is consequently a compromise rather than an
effective choice for both phases.

The paper also considers prefill/decode disaggregation, in which fixed GPU groups serve the two
phases. With constrained GPU capacity, this can imbalance resources: in the paper's 70B-on-eight-
40-GiB-GPUs example, a four-GPU split leaves prefill throughput more than six times higher than
decode throughput. Seesaw keeps the full GPU pool available to both phases and changes the model's
distribution instead of permanently dividing the cluster.

=== Method

==== Dynamic model re-sharding

Seesaw changes TP and PP between phases while keeping the data-parallel degree fixed. During
prefill, layers can be placed in pipeline stages; during decode, shards of every layer can instead
be spread across a tensor-parallel group. The transition re-shards both model weights and the KV
cache. Required weight shards are reloaded from CPU memory. For the KV cache, workers first write
their current shards to shared CPU memory, which acts as a complete intermediate representation, and
then read the shards required by the decode configuration.

==== Tiered buffering and scheduling

Re-sharding after every small batch would eliminate its gains, so Seesaw adds CPU memory as a second
KV-cache tier. It accumulates prefetched requests and swaps their KV caches to the CPU buffer. Once
the buffer fills, the system transitions to the decode configuration and runs a large decode batch,
progressively moving caches back to GPU memory. When the buffer drains, it switches back to prefill.
This transition-minimizing schedule amortizes model movement and improves model-weight loading
efficiency through batching.

==== Asynchronous data movement

Seesaw overlaps KV-cache transfers with computation where possible. Swap-out can proceed while later
prefill work runs, and per-worker background prefetchers perform swap-in before the associated
decode work is scheduled. The goal is for total time to approach the larger of computation and
transfer time rather than their sum. The evaluation allocates 80 GiB of CPU memory per GPU for the
buffering design.

=== Pros and Cons

==== Pros

- The phase-specific choice of PP and TP follows a clear systems observation, and the evaluation
  includes configurations where PP4 is best for prefill while TP4 is best for decode.
- All GPUs participate in both phases, which can use constrained hardware more effectively than a
  permanently disaggregated prefill/decode split.
- CPU-backed KV buffering, transition-aware scheduling, asynchronous transfers, and a bandwidth-
  aware layout directly address the overhead that a naive re-sharding implementation would incur.
- The reported results show 1.45 times geometric-mean speedup and up to 1.78 times speedup on A10
  systems, 1.29 times average and up to 1.52 times on L4 systems, and a 1.36 times overall average
  over the tested static vLLM configuration. Gains were reported across model sizes and ShareGPT and
  arXiv summarization workloads.

==== Cons

- The design targets throughput rather than interactive latency. Accumulating many prefills before
  decoding is not automatically suitable for time-to-first-token, inter-token-latency, or latency-
  SLO workloads.
- The optimization requires substantial host-memory capacity and adds CPU--GPU traffic; the reported
  evaluation uses 80 GiB of CPU memory per GPU.
- Benefits depend on the interconnect and workload mix. Faster NVLink reduces TP's prefill penalty,
  while workloads dominated almost entirely by prefill or decode leave less opportunity for a
  phase-specific switch. On eight NVLink-connected A100 GPUs, one reported ShareGPT improvement was
  13%, compared with larger gains on PCIe-connected systems.
- Reconfiguration is limited to TP and PP. Changing the data-parallel degree would alter model and
  KV-cache memory allocation and require additional data movement and implementation complexity.
- The main comparison is against vLLM 0.5.4; the supplied notes do not establish superiority over
  every major serving stack because several alternatives lacked the required PP support or were
  treated as behaviorally similar for this setting.
