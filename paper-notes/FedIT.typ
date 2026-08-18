== Towards Building the Federated GPT: Federated Instruction Tuning

=== Summary

*FedIT*@zhangBuildingFederatedGPT2024 is a way to instruction-tune an LLM using instruction data
distributed across many users or organizations *without collecting the raw instruction data in one
central place*.

The key combination is:

$ "Federated Learning" + "Parameter-Efficient Fine-Tuning" $

More specifically, the paper uses LoRA as the parameter-efficient fine-tuning mechanism. The
pretrained LLM is kept mostly frozen; each client trains only a small adapter on its own local
instruction data, and the server aggregates those adapters into a shared global adapter.

The conceptual contribution is therefore not a new LLM architecture. It is a *training framework for
decentralized instruction tuning*.

=== What problem do they want to address?

The paper starts from a conflict.

Instruction tuning benefits from:

$ "large quantity" + "high quality" + "high diversity" $

of instruction-response data. But much useful instruction data is produced by users or organizations
and may be sensitive, proprietary, expensive, or simply unavailable for centralized collection.

Traditional instruction tuning assumes something like:

$ D_1 + D_2 + D_3 + dots + D_N arrow "central dataset" arrow "fine-tune LLM" $

The authors want to replace that with

$ D_1, D_2, D_3, dots, D_N $

remaining on their respective clients, while still producing one improved shared model.

There is a second problem: *LLMs are too large for conventional federated learning*. Full
fine-tuning would mean every client has to train and communicate a huge number of parameters. The
authors therefore need a method that simultaneously solves

$ "data decentralization" $

and

$ "computation / communication cost". $

Their solution is federated learning for the first and LoRA for the second.

=== Methodology

This is really the core of the paper.

==== The whole method in one line

$
  "Select clients" arrow "local LoRA instruction tuning" arrow "aggregate LoRA updates" arrow "repeat"
$

The system has *one server and many clients*.

==== Global model

Let the pretrained LLM parameters be

$ w. $

The base model is shared among clients and remains frozen during local training.

The trainable part is a lightweight adapter:

$ Delta w. $

So conceptually the usable model is

$ w + Delta w. $

The important point is that clients do *not* repeatedly fine-tune all of $w$.

==== Each client owns private instruction data

Client $i$ has its own dataset:

$ D_i. $

For example:

$ D_1 = "writing instructions" $

$ D_2 = "question answering" $

$ D_3 = "summarization" $

The datasets may differ in task, language, domain, writing style, complexity, culture, and other
characteristics. The authors deliberately treat this heterogeneity as a central property of FedIT
rather than assuming every client has similarly distributed data.

==== Server selects clients

At communication round $k$, the server selects a subset of clients:

$ M_k subset.eq {1, dots, N}. $

Only those clients participate in that round.

This matters because in real federated systems not every device will always be available. The paper
also suggests that client selection could eventually consider the client's data characteristics and
computational resources rather than merely choosing arbitrary users.

==== Clients locally instruction-tune the adapter

The server sends the current global adapter

$ Delta w^(k) $

to the selected clients.

Each client freezes the base model $w$ and trains only the adapter using its own dataset $D_i$:

$ Delta w_i^(k+1) = op("InstructionTune")(Delta w^(k), D_i). $

The instruction data itself remains at the client.

So client $i$ transforms

$ (w, Delta w^(k), D_i) $

into

$ Delta w_i^(k+1). $

Only the model update needs to leave the client.

==== Why LoRA?

Instead of learning an arbitrary large update matrix $Delta W$, LoRA writes it as:

$ Delta W = B A, $

where $A$ and $B$ are low-rank matrices.

If

$ W in RR^(d times d), $

full fine-tuning potentially updates roughly

$ d^2 $

values.

With LoRA:

$ A in RR^(r times d), quad B in RR^(d times r), $

with

$ r << d. $

So the number of trainable values is approximately

$ 2 d r $

instead of

$ d^2. $

That is what makes local training and communication much more manageable. The authors choose LoRA
because they need parameter-efficient training inside the federated-learning loop.

So LoRA is not the main conceptual contribution. It is the *enabler* that makes FedIT practical
enough to study.

==== Server aggregates client adapters

The selected clients return

$ Delta w_1^(k+1), Delta w_2^(k+1), dots $

The server combines them:

$ Delta w^(k+1) = op("Aggregate")(({Delta w_i^(k+1)})_(i in M_k)). $

This produces a new *global adapter*.

In the simplest case this can be FedAvg-style averaging:

$ Delta w^(k+1) = sum_(i in M_k) (abs(D_i)) / (sum_(j in M_k) abs(D_j)) Delta w_i^(k+1). $

Then the next round starts.

So the complete algorithm is essentially:

$ Delta w^(0) $

$ arrow "select clients" $

$ arrow "each client trains" Delta w_i^(1) $

$ arrow "aggregate" Delta w^(1) $

$ arrow "select clients again" arrow "local training" arrow "aggregate" arrow dots $

until training ends. This is almost exactly Algorithm 1 in the paper.

==== The most interesting methodological idea: heterogeneity

This is the part worth paying particular attention to.

Normally in federated learning,

$ "heterogeneous client data" $

is regarded as a *problem*.

For example,

$ D_A = "medical" $

while

$ D_B = "creative writing". $

Their gradients may pull the shared model in very different directions. But instruction tuning needs
broad task diversity.

So the authors argue that

$ "client heterogeneity can also be useful training diversity" $

because each client effectively contributes different capabilities. They explicitly discuss
heterogeneity across task categories, languages, domains, complexity, ambiguity, emotional tone, and
culture.

That gives FedIT an interesting tension:

$ "heterogeneity" arrow cases("harder federated optimization", "richer instruction diversity") $

This is arguably the intellectually important part of the paper, more than the use of LoRA itself.

=== Pros and cons

==== Pros

+ *It unlocks decentralized instruction data.* Organizations or users do not need to construct a
  single centralized dataset before collaborating. Their instruction data remains locally stored.

+ *LoRA fits federated learning naturally.* Federated learning is communication-heavy; full LLM
  fine-tuning would make that problem much worse. Training and transmitting only adapters reduces
  both local computation and the amount of model state that must be exchanged.

+ *It provides a natural way to exploit distributed expertise.* A medical organization, legal
  organization, programmer, multilingual user, and so on can possess very different instruction
  distributions. FedIT provides a mechanism for those distributions to contribute to one shared
  model.

+ *It is modular.* The framework can conceptually swap different components:

  $ "LoRA" arrow "other PEFT method" $

  $ "FedAvg" arrow "better aggregation" $

  $ "random selection" arrow "intelligent client selection". $

  The paper explicitly presents better client selection, aggregation, personalization, and PEFT
  methods as directions for improving the framework.

==== Cons

===== “Data stays local” does not mean “privacy is solved”

This distinction is important.

FedIT avoids sending the *raw instruction dataset*, but clients still transmit learned updates:

$ Delta w_i. $

Those updates can potentially leak information about the local training text. The authors
specifically discuss attacks capable of recovering text from language-model gradients and mention
differential privacy and gradient pruning as possible defenses.

Therefore:

$ "Federated learning" != "formal privacy guarantee" $

FedIT needs additional privacy mechanisms if strong privacy is required.

===== Heterogeneity is simultaneously its strength and weakness

Suppose $Delta w_A$ learns medical reasoning, while $Delta w_B$ learns informal conversational
behavior. Simply averaging them does not guarantee that both capabilities combine cleanly.

Thus the same diversity FedIT wants to exploit creates an optimization problem:

$ "diverse useful knowledge" quad "vs." quad "conflicting local updates". $

The paper itself identifies personalization and heterogeneous client distributions as important
unresolved issues.

===== Aggregation is rather simplistic

At the heart of FedIT is:

$ Delta w_("global") = op("Aggregate")(Delta w_1, dots, Delta w_n). $

But there is no guarantee that simple parameter averaging is the best way to combine very different
skills.

For example, suppose

$ A = "legal expert", quad B = "Japanese speaker", quad C = "programmer". $

Averaging their adapters treats knowledge combination largely as a numerical optimization problem.

A more sophisticated system might use:

- task-aware aggregation,
- routing,
- adapter composition,
- mixture-of-experts,
- clustering similar clients, or
- personalized global/local adapters.

The paper largely leaves this open.

===== Communication is still a serious problem

LoRA makes communication smaller, but the basic FedIT loop still requires:

$ "download" arrow "train" arrow "upload" arrow "repeat many times". $

The authors themselves describe LLM federated learning as facing very large communication,
computation, and storage costs.

So PEFT reduces the problem; it does not remove it.

===== Malicious clients can poison the shared model

Because the server trusts client updates, an attacker could deliberately train on harmful or
manipulated instructions:

$ D_("malicious") arrow Delta w_("malicious") arrow "global aggregation". $
