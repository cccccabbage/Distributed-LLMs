== FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models <paper-fedex-lora>

=== Summary

FedEx-LoRA @singhalFedExLoRAExactAggregation2025 combines @background-federated-learning[Federated
  Learning] with @background-lora[LoRA: Low-Rank Adaptation] fine-tuning and addresses the
aggregation mismatch described in @issue-lora-aggregation-mismatch[LoRA Factor/Product Aggregation
  Mismatch]. It computes the missing residual between the exact average update and the product of
the averaged factors, then adds that residual to the frozen base model. The aggregated model then
coincides exactly with full-model federated averaging, while clients keep training only low-rank
adapters.

=== Issues Addressed

FedEx-LoRA addresses the @issue-lora-aggregation-mismatch[LoRA Factor/Product Aggregation Mismatch].
Naive federated LoRA averages the two adapter factors independently and adopts the product of the
averages as the update, which deviates from the exact average of the client updates through unwanted
cross-client terms. Averaging the dense products directly is not a way out either, because the exact
average need not fit inside a rank-$r$ adapter. The paper-specific challenge is to preserve exact
full-model federated averaging on the server while keeping the trainable adapters low-rank. The
method also reshapes what must be transmitted each round, connecting it to
@issue-communication-cost[Communication Cost and Synchronization], and the exchanged updates remain
subject to @issue-privacy-leakage[Privacy Leakage Beyond Data Locality].

=== Method

FedEx-LoRA splits the aggregated update into a low-rank part that stays in the LoRA factors and a
residual that is absorbed into the backbone.

At the start of a round, each client receives the backbone $W_0$ and the current factors $A$ and
$B$. During local training, $W_0$ remains frozen and only $A$ and $B$ receive gradients. Client $i$
then communicates its trained factors $A_i$ and $B_i$ to the server.

The server first averages the factors, $bar(A) = 1 / K sum_i A_i$ and $bar(B) = 1 / K sum_i B_i$,
which keep the dimensions and rank structure of ordinary LoRA parameters. It also forms the exact
target update

$
  Delta W_"exact" = 1 / K sum_(i = 1)^K B_i A_i.
$

The residual is the information lost when the factors are averaged independently:

$
  R = Delta W_"exact" - bar(B) bar(A) = 1 / K sum_i B_i A_i - bar(B) bar(A).
$

Rather than forcing $R$ into another rank-$r$ adapter, FedEx-LoRA merges it into the backbone,
$W_0^"new" = W_0 + R$, and keeps $A^"new" = bar(A)$ and $B^"new" = bar(B)$. The aggregated model is
then

$
  W^"new" = W_0^"new" + B^"new" A^"new" = W_0 + R + bar(B) bar(A) = W_0 + 1 / K sum_i B_i A_i = 1 / K sum_i W_i,
$

so the aggregation is exact and coincides with full-model federated averaging.

This backbone update does not break LoRA. "Frozen" refers to optimization: $W_0$ receives no
gradients during local training, but it may change between federated rounds. Each client still
optimizes only $A$ and $B$, so the number of trainable parameters is unchanged. Conceptually, $W_0$
serves as the current frozen backbone rather than the original pretrained weights forever.

The residual itself satisfies $op("rank")(R) = O(K r)$: higher rank than a single LoRA update, but
still governed by the client count and the LoRA rank. The source notes that it may potentially be
represented and communicated in factored form. When $K$ is large and exact residual transmission
becomes too expensive, FedEx-LoRA can truncate the singular value decomposition $R = U Sigma V^top$
to its largest $r'$ singular values, $R approx U_(r') Sigma_(r') V_(r')^top$. A smaller $r'$ lowers
communication cost but weakens exactness; only the untruncated variant retains the full residual
information.

=== Pros and Cons

==== Pros

- The aggregation is exact: the reconstructed global model equals $1 / K sum_i W_i$, removing the
  deviation that separate factor averaging introduces through cross-client terms.
- Local training stays low-rank: clients optimize only $A_i$ and $B_i$ against a frozen backbone, so
  LoRA's parameter efficiency is preserved and the number of trainable parameters does not grow.
- The core idea is simple: one residual computation merged into the base model, with no change to
  the LoRA training procedure itself.
- The high-rank exact average $1 / K sum_i B_i A_i$ never has to be compressed back into a rank-$r$
  adapter; the surplus information lives in the backbone instead.
- The truncated-SVD variant exposes a tunable trade-off between communication cost and aggregation
  accuracy through the chosen residual rank $r'$.

==== Cons

- The residual rank grows as $O(K r)$, so transmitting the exact residual becomes expensive with
  many clients; the exact scheme is best suited to cross-silo settings with relatively few clients.
- The server must form every product $B_i A_i$, construct $R$, and run an SVD when compression is
  used, which is more computation than plainly averaging factors.
- Communication can exceed standard federated LoRA, which exchanges only adapter matrices, because
  the residual must also be distributed.
- Truncation removes the exactness guarantee: with $R approx R_(r')$, the merged model $W_0 +
  R_(r') + bar(B) bar(A)$ no longer equals $W_0 + 1 / K sum_i B_i A_i$, forcing a choice between
  exactness and compression.
- Exact aggregation is not centralized-training equivalence: clients still run separate local
  optimization on potentially different data distributions, so the client-drift concerns of
  @issue-data-heterogeneity[Data Heterogeneity and Client Drift] remain.
- The formulation assumes compatible LoRA structures across clients; substantially different ranks
  or adapter configurations make aggregation less straightforward.
- Keeping data local is not a privacy guarantee, and FedEx-LoRA adds no protection such as secure
  aggregation or differential privacy against the leakage risks in @issue-privacy-leakage[Privacy
    Leakage Beyond Data Locality].
