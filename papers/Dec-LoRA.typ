== Decentralized Low-Rank Fine-Tuning of Large Language Models <paper-dec-lora>

=== Summary

Dec-LoRA is a serverless approach to collaborative, parameter-efficient LLM fine-tuning. Clients
keep their datasets local, train their own @background-lora[LoRA: Low-Rank Adaptation] factors, and
exchange those factors only with neighbors in a communication graph
@ghiasvandDecentralizedLowRankFineTuning2025. Rather than introducing a central aggregation server,
the method uses repeated local training and neighbor-weighted mixing.

The paper evaluates Dec-LoRA on BERT and LLaMA-2, including heterogeneous-data and quantized-base
settings. It reports performance comparable to centralized LoRA in its evaluated conditions and
proves convergence to a stationary point for non-convex, smooth objectives under its stated
assumptions @ghiasvandDecentralizedLowRankFineTuning2025.

=== Issues Addressed

Although @background-federated-learning[Federated Learning] keeps raw data local, its coordinator
can be a bottleneck and a single point of coordination or failure. Dec-LoRA replaces that
coordinator with direct, neighbor-to-neighbor communication.

The communication constraints summarized in @issue-communication-cost[Communication Cost and
  Synchronization] motivate exchanging only the low-rank factors of @background-lora[LoRA: Low-Rank
  Adaptation]. For rank $r$, this reduces the state transferred per adapted layer from a dense
update to roughly $O((d_1 + d_2) r)$.

The paper also addresses an optimization complication specific to LoRA. If client $i$ has factors
$A_i$ and $B_i$, separately averaging them generally does not produce the average dense update:

$
  (1 / n sum_i B_i) (1 / n sum_i A_i) != 1 / n sum_i B_i A_i.
$

Its analysis therefore works directly with the two factors. The local-update drift, imperfect
consensus, and graph-connectivity trade-offs are discussed in @issue-data-heterogeneity[Data
  Heterogeneity and Client Drift]. Exchanged adapter parameters retain the privacy risks summarized
in @issue-privacy-leakage[Privacy Leakage Beyond Data Locality].

=== Method

Each of $n$ clients holds the same frozen base model, a private local dataset, and its own LoRA
factors $A_i$ and $B_i$. In a communication round, client $i$ takes $K$ local stochastic-gradient
steps on its data, updating only those factors. The base-model weights remain frozen.

Clients then exchange their factors with neighboring clients and apply a graph mixing matrix $Q$:

$
  A_i^(t+1) = sum_j q_(i j) A_j,
$

$
  B_i^(t+1) = sum_j q_(i j) B_j.
$

The resulting factors seed the next local-training round. A sparse topology, such as a ring, lowers
each client's communication degree but spreads information more slowly; more connected graphs
accelerate consensus at higher per-round communication cost. The convergence result has order
$O(1 / sqrt(T))$ and is formulated through gradients with respect to the LoRA factors
@ghiasvandDecentralizedLowRankFineTuning2025.

The authors also evaluate Dec-LoRA with a 4-bit quantized frozen base model, relating it to the
memory-saving setting described in @background-qlora[QLoRA: Quantized LoRA].

=== Pros and Cons

==== Pros

- No central parameter server is required, which removes a central coordination dependency.
- Clients exchange small LoRA factors rather than full model weights, reducing communication.
- Raw training data stays on each client.
- The reported experiments show performance comparable to centralized LoRA in the evaluated BERT,
  LLaMA-2, heterogeneous-data, and quantized-base settings.
- The paper provides a convergence analysis tailored to LoRA's bilinear parameterization.

==== Cons

- Its evaluated heterogeneous-data and graph settings remain subject to the client-drift and
  communication trade-offs described in @issue-data-heterogeneity[Data Heterogeneity and Client
    Drift].
- The method does not add a defense against the leakage risks from exchanged parameters described in
  @issue-privacy-leakage[Privacy Leakage Beyond Data Locality], or against malicious clients.
- The analysis and experiments use controlled communication graphs and do not fully address client
  churn, unreliable links, asynchronous updates, heterogeneous hardware, or adversarial behavior.
