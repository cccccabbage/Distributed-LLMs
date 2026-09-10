== Cache Locality versus Load Balancing in Distributed Serving <issue-cache-locality-load>

Distributed LLM serving can reuse the KV-cache prefixes described in
@background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache]. Because that state
lives on particular workers, routing is no longer a choice among interchangeable replicas. Sending a
request to a worker that already holds a matching prefix can avoid repeated prefill, while sending
it elsewhere may recompute the same tokens or transfer the cache
@srivatsaPrebleEfficientDistributed2025 @huDeepServeServerlessLarge2025.

Pure load balancing ignores this state. Requests that share a prefix can be scattered across
workers, each of which then recomputes and stores a separate copy. Pure locality routing has the
opposite failure: a popular prefix concentrates work on the workers that hold it, creating hotspots
even when other workers are idle. Replicating a popular cache can spread future reuse, but consumes
additional KV-cache memory and may evict other useful prefixes.

A practical scheduler therefore jointly considers cache affinity and current load, and may also
weigh the cost of transferring cached state against recomputing it, the memory cost of replication,
and estimates of remaining work such as expected output length. Those estimates can be wrong, so the
routing decision remains approximate.

Maintaining a global view of cache locations, prefix popularity, and worker load simplifies these
decisions, but a centralized scheduler can itself become a coordination bottleneck as the cluster
grows. Larger deployments may need hierarchical or partitioned scheduling. This issue is about
request routing given existing KV-cache state; it is distinct from the device-placement problem in
@issue-inference-resource-heterogeneity[Distributed-Inference Resource Heterogeneity and Model
  Placement].
