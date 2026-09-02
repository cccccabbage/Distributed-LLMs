== Preble: Efficient Distributed Prompt Scheduling for LLM Serving <paper-preble>

=== Summary

Preble@srivatsaPrebleEfficientDistributed2025 is a distributed LLM-serving scheduler designed to
reuse shared prompt-prefix KV caches across GPUs while avoiding cache-induced load hotspots. It
targets long-context workloads in which many requests share documents, system prompts, tool
descriptions, or conversation history, so recomputing the prefill phase for every request is costly.
Its central E2 (Exploitation and Exploration) policy selects between directing a request to an
existing cache and distributing work to a less-loaded GPU. The evaluated results indicate that its
benefits are strongest on workloads with substantial prefix sharing.

=== Issues Addressed

Preble builds on the prefill, decode, and cache-locality trade-offs described in
@background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache]. Its particular
workload contains long inputs and short outputs, so repeated prefill is costly. Although a cached
shared prefix can eliminate much of this repeated work, a conventional load balancer may route
requests with the same prefix to different GPUs. Each GPU then recomputes and stores a separate
copy, losing cache locality.

Always routing a request to the GPU holding its longest cached prefix is also insufficient: a
popular prefix can turn that GPU into a hotspot. Preble therefore frames distributed prompt
scheduling as a trade-off between cache reuse, computational load, cache capacity and eviction, and
request fairness. This is a serving-efficiency problem rather than a federated-training or formal
data-privacy mechanism.

=== Method

Preble uses a global scheduler above a local scheduler on each GPU. The global scheduler maintains a
radix tree of prompt prefixes, including their lengths, cached locations, and recent popularity. For
an incoming prompt, it finds the longest matching prefix and uses cache-location and GPU-load
information to select a destination. Each local scheduler executes and batches requests, manages KV
cache eviction and chunked prefill, and enforces the local priority policy.

E2 compares the reusable shared-token portion $S$ of a prompt with its uncached portion $U$. When
reuse is valuable, exploitation routes the request to a GPU with a long matching prefix to avoid
prefill. When the cache saving is small relative to uncached work, exploration considers other GPUs
to spread load and may create another useful cache replica. During exploration, the scheduler
estimates a GPU's cost from existing prefill and decode load, the potential cost of evicting useful
cached entries, and the computation needed for the new request, then selects the lowest-cost option.

The system dynamically shifts new requests away from overloaded GPUs and autoscales a popular prefix
by replicating it to multiple GPUs. This makes future cache reuse available at several destinations,
at the expense of KV-cache memory. To prevent requests with weak cache matches from starving, local
scheduling groups requests by cache-hit percentage: high-hit requests receive priority, while
lower-priority groups are periodically scheduled.

=== Pros and Cons

==== Pros

- The scheduler directly combines prefix-cache locality with load balancing, rather than treating
  them as independent decisions.
- It is well matched to long-context applications with repeated prefixes, including document QA,
  retrieval-augmented generation, tool use, and multi-turn conversations.
- Dynamic load shifting and prefix replication allow the system to respond to changing prefix
  popularity instead of assuming a static workload.
- The reported evaluation shows substantial average and tail-latency improvements over the evaluated
  distributed SGLang baselines, particularly when prompts share large prefixes.
- Accounting for cache-eviction cost and scheduling fairness makes the policy more practical than
  simply preferring the longest cache hit.

==== Cons

- Benefits depend on substantial prefix sharing. Nearly unique prompts offer little cache reuse and
  therefore less advantage over ordinary load balancing.
- KV-cache reuse mainly reduces prefill work; decode-heavy workloads with long generated outputs may
  benefit less.
- Cost estimates based on token counts, recent load, cache state, and expected output length can be
  inaccurate for unpredictable requests, producing suboptimal routing decisions.
- Maintaining a global radix tree, cache-location mappings, popularity information, and load
  statistics increases coordination complexity; the centralized scheduler may itself become a
  scalability concern at much larger cluster sizes.
- Prefix autoscaling trades load balance for GPU memory: replicas consume KV-cache capacity and can
  evict other reusable prefixes.
- The supplied notes describe evaluations on selected workloads and configurations; performance in
  heterogeneous, multi-tenant, geographically distributed, or rapidly changing production
  deployments remains to be established.
