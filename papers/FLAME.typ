== FLAME: Federated Fine-Tuning LLMs Through Adaptive SMoE <paper-flame>

=== Summary

FLAME@leFLAMEFederatedFineTuning2025 is a federated fine-tuning framework for Sparse
Mixture-of-Experts (SMoE) language models with clients that have different computational resources.
Instead of compressing the global LoRA adapter to a lower rank for weaker clients, FLAME keeps the
LoRA rank unchanged and varies the number of active experts. Stronger clients can execute more
experts, while weaker clients execute fewer, reducing computation in the base model as well as in
the adapter path. The framework also introduces a learnable rescaling factor for the output of a
partial expert set and activation-aware aggregation so that an expert's update is weighted more
strongly by clients that actually trained that expert.

The paper reports that reducing active experts can substantially lower computation. In its reported
configuration, reducing the active-expert count from 8 to 1 reduces estimated FLOPs from 342.8B to
158.0B, approximately 54 percent, while retaining full-rank LoRA matrices. Across its evaluated
computational settings, FLAME outperforms the resource-adaptive federated baselines considered by
the paper.

=== Issues Addressed

FLAME addresses @issue-resource-heterogeneity[Resource Heterogeneity and Configuration Adaptation]
by assigning clients different compute configurations. A common configuration must otherwise be
small enough for the weakest client and can leave stronger clients underused, while synchronous
training can be slowed by stragglers.

Earlier resource-adaptive LoRA methods reduce a weak client's LoRA rank. That reduces adapter
computation, but most computation still comes from the base language model. Rank reduction can also
discard information when a global adapter is approximated at a lower rank. FLAME moves the
resource-dependent adjustment to the MoE execution path: the client changes how many experts are
executed while retaining the full LoRA rank.

Partial expert activation creates a second problem. Executing fewer experts can change the magnitude
of the combined MoE output, so a client-specific expert count may destabilize training or alter the
representation scale. FLAME addresses this with a learnable rescaling factor rather than assuming
that a fixed inverse relationship with the number of active experts is sufficient.

Finally, different clients activate different experts with different frequencies. Dataset size alone
is therefore an incomplete proxy for how much a client trained a particular expert. FLAME uses
per-expert activation frequencies during aggregation. Its communication remains subject to the
trade-offs described in @issue-communication-cost[Communication Cost and Synchronization], and
keeping data local does not itself resolve the risks described in @issue-privacy-leakage[Privacy
  Leakage Beyond Data Locality].

=== Method

==== Adaptive Sparse Mixture-of-Experts Execution

An MoE layer contains experts $E_1, E_2, ..., E_M$. A router scores the experts for each token and
selects a Top-$k$ subset to execute. FLAME assigns client $i$ an active-expert count $k_i$ based on
its available resources. For example, a high-capability client may use 8 experts, a medium client 4,
a low-capability client 2, and a very limited client 1. The router still chooses which experts are
selected; FLAME controls how many can be selected.

All clients retain the same LoRA rank. For expert $j$, with frozen or shared expert weight $W^j$ and
LoRA factors $A_i^j$ and $B_i^j$, the client-side expert transformation is represented as

$ W^j x + A_i^j B_i^j x. $

Thus, resource adaptation changes the amount of the base MoE computation rather than repeatedly
compressing the global adapter for weaker clients.

==== Learnable Rescaling

If the usual model activates $k$ experts but client $i$ activates only $k_i$, a simple fixed
correction might use $s_i = k / k_i$. FLAME instead learns a scalar rescaling factor $s_i$. The
client output is approximately

$ h = s_i sum_j R_i(x, k_i)^j (W^j x + A_i^j B_i^j x), $

where $R_i(x, k_i)^j$ is the router weight for expert $j$. The factor scales the combined expert
output; it is not merely an additional weight applied to the LoRA update.

==== Activation-Aware Aggregation

During local training, client $i$ records how frequently each expert $j$ is activated. With $a_i^j$
denoting the number of activations and $S_i$ the relevant number of local training steps, the
activation frequency is

$ f_i^j = a_i^j / S_i. $

A high value indicates that the client trained that expert frequently, while a value near zero
indicates little or no training for it. FLAME defines an expert-specific aggregation weight

$ gamma_i^j = (f_i^j)^t |D_i| $.

where $|D_i|$ is the client's dataset size and $t$ is a temperature or control parameter. Larger
activation frequency therefore gives the client more influence over that expert's global update. If
$t = 0$, the frequency term becomes neutral and the weighting is similar to dataset-size-based
FedAvg. Increasing $t$ emphasizes differences in expert usage more strongly.

The resulting workflow is: determine each client's resource-dependent expert count, fine-tune the
full-rank LoRA factors locally, record expert activation frequencies, and aggregate each expert with
activation-aware weights.

=== Pros and Cons

==== Pros

- Varying active experts reduces computation in the base MoE model. The reported 8-to-1 expert
  configuration lowers estimated FLOPs from 342.8B to 158.0B, or approximately 54 percent.
- Client-specific expert counts directly match the amount of executed model computation to
  heterogeneous resource budgets and let better-provisioned clients use more capacity.
- Keeping the LoRA rank unchanged avoids the information loss associated with repeatedly
  approximating a global adapter at a lower rank for weaker clients.
- Activation-aware aggregation accounts for which client actually trained each expert, which is a
  better fit for sparse expert usage than dataset-size weighting alone.
- The learnable rescaling factor gives the model a mechanism to compensate for output-scale changes
  caused by partial expert activation.

==== Cons

- FLAME depends on an MoE architecture with a controllable active-expert count and cannot be
  directly applied to an ordinary dense Transformer.
- Lower active-expert counts primarily reduce computation, not necessarily memory. A client may
  still need access to the full expert set, full-rank LoRA parameters, and routing parameters.
- Keeping full-rank adapters can leave communication costs higher than methods that transmit or
  maintain compressed adapters, so the method mainly targets compute heterogeneity rather than the
  combined compute, memory, and communication problem.
- The reported evaluation uses OLMoE and a limited set of instruction-tuning datasets. Validation on
  much larger MoE models, physical edge devices, larger client populations, longer federated runs,
  and production-scale environments remains necessary.
- The rescaling ablations do not show uniform improvements in every configuration. The more clearly
  central contributions are adaptive expert activation and activation-aware aggregation.
- The method does not provide formal protection against update leakage, malicious clients, or
  poisoned expert updates. Data locality should therefore not be treated as a privacy or integrity
  guarantee.
