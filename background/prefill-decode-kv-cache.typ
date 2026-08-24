== Prefill, Decode, and the Key-Value Cache <background-prefill-decode-kv-cache>

Autoregressive LLM inference has two distinct phases. During *prefill*, the model processes the
input prompt and produces the attention key-value (KV) cache. During *decode*, the model repeatedly
generates one output token at a time, reusing the accumulated cache for the prompt and preceding
output tokens. Prefill can process many prompt tokens together and is commonly compute-intensive;
decode exposes less token-level parallelism and is commonly constrained by memory bandwidth.

The KV cache avoids recomputing attention state for tokens already processed, but consumes device
memory and grows with the prompt and generated sequence. Reusing a cached prompt prefix can
substantially reduce repeated prefill work. In a distributed serving system, cache reuse is
therefore also a placement problem: routing a request to the cache's current location improves
locality, while replicating a popular cache consumes additional memory and can concentrate load.

Many systems colocate prefill and decode on one replica, keeping its cache local. Disaggregated
systems place the phases on different replicas to provision them independently or reduce
prefill--decode interference. They must then transfer the KV cache between the replicas. The size of
that transfer and the available network capacity can become a bottleneck, so an effective design
must jointly consider phase capacity, cache memory, and communication topology.
