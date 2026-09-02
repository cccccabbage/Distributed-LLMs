== Towards Building the Federated GPT: Federated Instruction Tuning <paper-fedit>

=== Summary

FedIT is a framework for instruction-tuning a large language model using instruction data
distributed across users or organizations without centralizing the raw data
@zhangBuildingFederatedGPT2024. It combines @background-federated-learning[Federated Learning] with
@background-lora[LoRA: Low-Rank Adaptation]: clients train lightweight adapters locally, and a
server aggregates those adapters into a shared global adapter. Its contribution is a decentralized
training framework rather than a new language-model architecture.

=== Issues Addressed

Instruction tuning benefits from large, diverse, high-quality instruction-response collections, but
useful data may be sensitive, proprietary, or unavailable for centralized collection. FedIT keeps
that data at its source while allowing it to contribute to a shared model.

FedIT addresses the communication constraints summarized in @issue-communication-cost[Communication
  Cost and Synchronization] by training and transmitting adapters rather than all model parameters.
Clients still exchange those adapters over multiple rounds.

FedIT treats client task, language, domain, and style differences as both a source of useful
instruction diversity and an optimization difficulty; see @issue-data-heterogeneity[Data
  Heterogeneity and Client Drift].

Finally, client updates retain the privacy risks summarized in @issue-privacy-leakage[Privacy
  Leakage Beyond Data Locality]. Malicious participants may also submit poisoned updates, so privacy
and robustness mechanisms are needed when those risks are in scope.

=== Method

FedIT uses one server and many clients. Let the pretrained, frozen model parameters be $w$, and let
the trainable global adapter at communication round $k$ be $Delta w^(k)$. Client $i$ owns private
instruction data $D_i$.

1. The server selects a subset of clients $M_k$ and sends them the current adapter.
2. Each selected client freezes $w$ and instruction-tunes only its adapter on $D_i$:

  $
    Delta w_i^(k+1) = op("InstructionTune")(Delta w^(k), D_i).
  $

3. Clients return their adapter parameters. The server combines them into $Delta w^(k+1)$ and
  repeats the process.

With data-size-weighted FedAvg-style aggregation, the update is

$
  Delta w^(k+1) = sum_(i in M_k) (abs(D_i)) / (sum_(j in M_k) abs(D_j)) Delta w_i^(k+1).
$

LoRA represents a dense layer update as $Delta W = B A$, where $A in RR^(r times d)$,
$B in RR^(d times r)$, and $r << d$. This changes the trainable parameter count from approximately
$d^2$ to $2 d r$ for a square $d times d$ matrix, making local optimization and update exchange more
manageable. FedIT treats client heterogeneity as both an optimization difficulty and a potential
source of useful instruction diversity. The paper leaves more advanced selection, aggregation,
personalization, and PEFT choices as extensions of the framework @zhangBuildingFederatedGPT2024.

=== Pros and Cons

==== Pros

- The framework enables collaboration without collecting raw instruction datasets in one place.
- Adapter-only training aligns well with the compute and communication constraints of federated LLM
  fine-tuning.
- Distributed clients can contribute different domains, languages, and task expertise to one model.
- The design is modular: client selection, PEFT, aggregation, and personalization components can be
  replaced or improved.

These advantages describe the framework and the motivation supported by the paper; they do not by
themselves establish formal privacy or robust convergence.

==== Cons

- The framework does not resolve the client-drift and aggregation challenges described in
  @issue-data-heterogeneity[Data Heterogeneity and Client Drift]; simple parameter averaging does
  not guarantee clean composition of distinct skills.
- LoRA reduces transmitted state, but clients still download, train, and upload across repeated
  rounds; the relevant communication trade-offs remain as described in
  @issue-communication-cost[Communication Cost and Synchronization].
- A malicious client can submit a poisoned adapter update. The framework does not itself provide a
  complete defense against poisoning or backdoors.
- The supplied paper discussion presents improved aggregation, client selection, personalization,
  and privacy protection largely as open directions rather than resolved components.
