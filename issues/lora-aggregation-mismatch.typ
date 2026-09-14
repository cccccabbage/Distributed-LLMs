== LoRA Factor/Product Aggregation Mismatch <issue-lora-aggregation-mismatch>

In federated @background-lora[LoRA: Low-Rank Adaptation] fine-tuning, each client trains adapter
factors on top of a shared frozen base model. For client $i$, the effective update to an adapted
layer is the product

$
  Delta W_i = B_i A_i,
$

where $B_i$ and $A_i$ are the trained low-rank factors. Because the base model is shared, exact
federated averaging of the client models corresponds to averaging these dense updates,

$
  Delta W = 1 / K sum_(i = 1)^K B_i A_i,
$

over the $K$ participating clients.

The natural implementation of that average in factor form is to average each factor separately. With
$bar(B) = 1 / K sum_i B_i$ and $bar(A) = 1 / K sum_i A_i$, this yields the update $bar(B) bar(A)$.
Expanding the product shows why it deviates from the intended average:

$
  bar(B) bar(A) = (1 / K sum_i B_i) (1 / K sum_j A_j) = 1 / K^2 sum_i sum_j B_i A_j.
$

The double sum mixes the desired terms $B_i A_i$ with cross-client terms $B_i A_j$ for $i != j$,
pairs of factors that no single client trained together. In general the average of products does not
factor into a product of averages, so the naive scheme does not reproduce the exact average update
@liuHLoRAEfficientFederated2025 @singhalFedExLoRAExactAggregation2025.

Averaging the products $B_i A_i$ directly removes the cross-client terms but collides with LoRA's
representation. Each product has rank at most $r$, while their sum can reach rank $K r$, so the
exact average need not fit inside a rank-$r$ adapter. Raising the rank after every aggregation, or
keeping every client's factors, erodes the parameter and communication savings that motivate
low-rank adapters in the first place. This creates a recurring trade-off between staying in a
compact low-rank parameterization and representing a dense aggregate exactly
@singhalFedExLoRAExactAggregation2025.

The mismatch appears whenever multiple clients' LoRA factors must be combined into one shared model,
in centralized-server federated learning with heterogeneous client ranks
@liuHLoRAEfficientFederated2025, in exact reconstruction of full-model federated averaging
@singhalFedExLoRAExactAggregation2025, and in serverless settings where neighbors mix factors
directly @ghiasvandDecentralizedLowRankFineTuning2025. Approaches differ in where they land between
factor-space averaging, dense reconstruction and refactorization, and absorbing a residual into the
base model; their specifics belong to the respective paper notes. Communication implications are
covered in @issue-communication-cost[Communication Cost and Synchronization], and optimization
effects of imperfect consensus in @issue-data-heterogeneity[Data Heterogeneity and Client Drift].
