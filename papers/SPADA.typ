== SPADA: Secure, Performant, and Distributed LLM Inference <paper-spada>

=== Summary

SPADA@chuSPADASecurePerformant2025 is a system architecture for secure distributed large language
model (LLM) inference. It addresses the full inference path rather than protecting only the user's
prompt: distributed nodes, model execution, network communication, and the key-value (KV) cache are
all included in the intended security boundary. Its four main components are Trusted Execution
Environments (TEEs), the Decentralized Trust Establishment Protocol (DTEP), secure communication,
and secure KV-cache transfer.

The architecture is intended for inference spread across multiple GPUs, machines, or cloud servers,
including cloud and multi-tenant settings. DTEP and mutual remote attestation establish trust
between nodes before sensitive data is sent. In-enclave Diffie-Hellman key establishment then
provides session keys for protected communication. Each node executes sensitive inference work in a
TEE, while in-TEE TLS protects data up to the trusted execution boundary. A binary protocol,
zero-copy transfer, padding, and burst aggregation target communication overhead and metadata
leakage. KV-cache fragments are encrypted and transferred with integrity and replay protections,
then delta encoding and prefetching reduce the cost and latency of moving cache state.

The paper's central contribution is architectural. The available note characterizes it as primarily
a design paper with limited extensive benchmark evidence, so the proposed performance optimizations
should be read as design goals and claims rather than as conclusions established by a broad
evaluation.

=== Issues Addressed

SPADA targets privacy risks in distributed inference, including exposure of prompts, embeddings,
intermediate model states, and KV-cache contents. These risks are an instance of
@issue-privacy-leakage[Privacy Leakage Beyond Data Locality]. The architecture treats the KV cache
as sensitive state because it is derived from user context and may carry information about prompts,
conversation history, and system prompts. This concern is specific to the cache movement required by
distributed inference and builds on the prefill, decode, and cache relationship described in
@background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache].

Distributing inference helps address the resource and scalability demands of large models and long
contexts, but it creates dependence on communication between nodes. SPADA therefore addresses the
bandwidth, copying, latency, and traffic-metadata concerns summarized in
@issue-communication-cost[Communication Cost and Synchronization]. It also applies to
@issue-inference-resource-heterogeneity[Distributed-Inference Resource Heterogeneity and Model
  Placement], since a deployment may span multiple machines or cloud servers with different
available resources and confidential-computing capabilities.

The security problems are layered. A node needs evidence that its peer is an authorized node running
expected trusted software. The host operating system, hypervisor, cloud provider, or another tenant
may also be able to inspect ordinary application memory. Finally, the network can be used to read,
modify, replay, or analyze traffic. SPADA addresses these problems through attestation, TEE
isolation, protected transport, and cache-specific protection. These mechanisms reduce the stated
attack surface, but data locality or encryption alone is not presented as a formal guarantee that
all information leakage is eliminated.

=== Method

SPADA organizes its design into four interacting layers.

==== Trust establishment with DTEP

DTEP, the Decentralized Trust Establishment Protocol, uses TEE-backed capabilities to establish
trust between distributed nodes. Before Node A sends sensitive information to Node B, B produces a
hardware-backed attestation report containing evidence about its TEE, the code and configuration
running inside it, and its enclave measurement. A verifies the report, and B performs the same
verification for A, making the attestation mutual. After both reports are accepted, the nodes
perform an in-enclave Diffie-Hellman key exchange. The resulting symmetric session key remains
inside the trusted environments and protects subsequent communication.

The protocol is decentralized in the sense that it does not require one SPADA-specific central trust
server to approve every relationship. This supports elastic clusters, cross-cloud deployments, and
nodes joining or leaving dynamically. It is not completely trustless, because the attestation
process still depends on the security and attestation infrastructure of the underlying TEE platform.

==== TEE-protected execution

SPADA places the model execution pipeline and sensitive states within a TEE. The intended protected
contents include user input, Transformer computations, attention state, generation history, the KV
cache, and session keys. Under the TEE security model, the untrusted host can run the enclave but
should not directly inspect those contents. The note discusses technologies such as Intel SGX and
AMD SEV as examples of the trusted-computing platforms involved.

TEE execution introduces memory, I/O, enclave-transition, paging, and programming-model costs. The
proposed mitigations include quantized operators, activation checkpointing, enclave-local cache
management, batched enclave calls, and asynchronous execution to reduce memory pressure and increase
overlap and parallelism.

==== Secure communication

SPADA terminates TLS inside the TEE, keeping plaintext and key material within the trusted boundary
before encrypted data reaches the network. The design mentions enclave-compatible TLS libraries such
as WolfSSL and Rustls-TEE. It also uses fixed-length headers, aligned buffers, and binary messages
instead of more expensive serialization formats. Zero-copy transfer avoids unnecessary intermediate
copies between enclave and network buffers, which is relevant for large intermediate states.

Encryption does not hide packet size, communication frequency, or timing. To make traffic analysis
more difficult, SPADA proposes padding and burst-mode aggregation. These measures address some
network metadata leakage while preserving the goal of keeping communication efficient.

==== Secure KV-cache transfer

KV-cache fragments are encrypted with the session keys established after attestation. Each
transmitted fragment also carries a nonce, sequence number, and message authentication code (MAC).
Encryption hides cache contents, the MAC detects modification, the nonce provides cryptographic
uniqueness, and the sequence number helps detect ordering and replay problems. Consequently, a
recorded cache update should not be replayable later as though it were new.

To avoid repeatedly sending the entire KV state, SPADA proposes delta encoding, sending only newly
changed KV information after an initial state. It also describes token-level prediction and
prefetching of relevant cache blocks. Computing the current token can overlap with fetching cache
data needed soon, reducing the wait that would result from strictly sequential computation and cache
transfer.

=== Pros and Cons

==== Pros

- SPADA presents an end-to-end architecture that considers nodes, execution, network communication,
  and KV caches together rather than treating security as simply adding TLS to distributed
  inference.
- Treating the KV cache as sensitive is well matched to distributed LLM serving, where cache state
  can encode user context and may be transferred or reused across machines.
- Hardware-backed mutual attestation provides a stronger basis for node trust than relying only on
  machine identities, and decentralized establishment fits elastic or multi-domain clusters.
- TEE execution, in-TEE TLS, encrypted cache fragments, nonces, MACs, and sequence numbers address
  distinct parts of the stated threat model, including privileged host software, network
  observation, modification, and replay.
- The design is performance-aware. Binary messages, zero-copy transfer, padding and burst
  aggregation, delta encoding, and prefetching target communication, metadata, bandwidth, and
  latency costs.
- The components are modular system mechanisms and do not require a new LLM architecture or decoding
  algorithm.

==== Cons

- TEE resource, memory, I/O, and execution constraints remain a substantial practical challenge for
  large model weights, KV caches, GPU computation, and memory bandwidth. Quantization,
  checkpointing, batching, and asynchronous execution manage this tension but do not remove the
  underlying hardware limitations.
- Security depends on the TEE implementation and its attestation mechanism. Vulnerabilities in the
  hardware, firmware, or attestation stack weaken the architecture's security assumptions.
- TEEs do not automatically eliminate timing, memory-access, traffic, or microarchitectural side
  channels. SPADA's use of padding and burst aggregation itself reflects that encryption alone is
  insufficient for hiding all network metadata.
- The complete design adds a TEE runtime, remote attestation, DTEP, key management, in-TEE TLS, KV
  encryption, nonce and sequence management, delta encoding, and prefetching. This increases
  implementation, debugging, deployment, and hardware-compatibility complexity, while security
  protocols are sensitive to implementation mistakes.
- Heterogeneous TEE support remains unresolved because platforms such as Intel SGX and AMD SEV
  differ in threat models, APIs, memory models, and attestation procedures. The supplied note
  identifies broader heterogeneous-TEE support as future work.
- The available note characterizes the paper as primarily an architecture/design paper with limited
  extensive benchmark evidence. It does not establish production-scale overhead, scalability,
  cache-transfer savings, or portability across different TEEs through a broad evaluation. Results
  comparing ordinary distributed inference and TLS, DTEP establishment time, encrypted and
  unencrypted cache transfer, delta encoding, prefetching, cluster size, context length, and TEE
  platform would be needed to assess those claims more fully.
