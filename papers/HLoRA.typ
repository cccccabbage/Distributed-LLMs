== HLoRA: Efficient Federated Learning System for LLM Heterogeneous Fine-Tuning <paper-hlora>

=== Summary

HLoRA@liuHLoRAEfficientFederated2025 is a federated fine-tuning method that lets clients train
@background-lora[LoRA: Low-Rank Adaptation] adapters at different ranks. Its central idea is to
reconstruct each client's dense LoRA update before aggregation, then factorize the aggregated update
with singular value decomposition (SVD) to redistribute rank-appropriate adapters. This avoids
averaging LoRA factors independently, which does not in general equal averaging their products.

On RoBERTa-large fine-tuned for MRPC, RTE, and QQP with 100 simulated clients, the paper reports
that its heterogeneous configuration outperforms its homogeneous HLoRA and naive federated-LoRA
baselines, though centralized LoRA remains more accurate.

=== Issues Addressed

HLoRA uses @background-federated-learning[Federated Learning] to fine-tune on distributed data
without centralizing raw training examples. Its exchanged adapter parameters retain the privacy
risks summarized in @issue-privacy-leakage[Privacy Leakage Beyond Data Locality]; the method does
not add a protection mechanism.

The paper instantiates @issue-resource-heterogeneity[Resource Heterogeneity and Configuration
  Adaptation] through a client-specific LoRA rank. Allowing ranks to differ creates factor matrices
with incompatible shapes. It also exposes an aggregation bias: independently averaging client
factors combines factors that no single client jointly trained.

=== Method

Client $k$ receives and locally trains LoRA factors $B_k$ and $A_k$ at rank $r_k$, while the
pretrained base-model weights remain frozen. The client's effective layer update is

$
  Delta W_k = B_k A_k.
$

After local training, the server reconstructs this dense update for every client and performs
sample-weighted aggregation,

$
  Delta W_("global") = sum_(k=1)^K (n_k / n) B_k A_k,
$

where $n_k$ is client $k$'s sample count and $n$ is the total participating sample count. Every
reconstructed update has the layer's dense shape regardless of $r_k$. In contrast, separately
averaging the factors would produce

$
  (sum_k p_k B_k) (sum_k p_k A_k),
$

which includes cross-client products such as $B_1 A_2$ and therefore generally differs from
$sum_k p_k B_k A_k$.

The server decomposes the global update as $Delta W_("global") = U Sigma V^T$. For a client with
target rank $r_k$, it retains the leading $r_k$ singular components and forms, for example,
$B'_k = U_(r_k)$ and $A'_k = Sigma_(r_k) V_(r_k)^T$. The resulting adapter is a rank-$r_k$
approximation to the shared dense update and initializes that client's next local-training round.
Thus, stronger and weaker clients receive the same global information at different capacities.

=== Pros and Cons

==== Pros

- Reconstructing and averaging $B_k A_k$ gives the intended weighted average of client updates and
  avoids untrained cross-client factor combinations.
- Different ranks let clients trade adapter capacity against their available compute, memory, and
  communication resources while participating in one federation.
- Clients retain LoRA's parameter-efficient local training because the frozen base model is not
  fine-tuned.
- In the reported MRPC, RTE, and QQP experiments, heterogeneous HLoRA is the strongest evaluated
  federated-LoRA variant: it attains 87.1, 86.1, and 88.4, respectively, compared with 84.0, 78.3,
  and 83.7 for naive federated LoRA.

==== Cons

- Each round requires the server to materialize dense layer updates, aggregate them, run SVD, and
  generate multiple-rank adapters. The paper's experiments do not establish this cost at
  multi-billion-parameter LLM scale.
- The method supports different ranks but does not provide a sophisticated policy to assign or adapt
  ranks from device memory, speed, bandwidth, energy, data volume, or utility.
- Evaluation uses RoBERTa-large, three language-understanding datasets, and simulated clients on GPU
  servers. Results may not transfer directly to modern very large LLMs or physically diverse client
  fleets.
- Centralized LoRA is still more accurate in the reported experiments: its MRPC, RTE, and QQP scores
  are 90.2, 87.4, and 91.6, respectively. HLoRA is therefore most relevant where pooling data for
  centralized training is infeasible or inappropriate.
