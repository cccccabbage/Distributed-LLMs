== Privacy Leakage Beyond Data Locality <issue-privacy-leakage>

Federated and split learning keep raw records at their original holders, but that data locality is
not by itself a privacy guarantee. Training and inference can still reveal information through the
messages needed for collaboration: individual model updates, gradients, intermediate activations,
labels, predictions, and metadata such as participation or timing. The relevant risk depends on the
threat model, the model and message design, batch size, auxiliary information available to an
observer, and whether participants are trusted.

In federated learning, an observer who sees an individual gradient or update may attempt to infer
properties of the contributing data or reconstruct examples. Deep Leakage from Gradients
demonstrated reconstruction from shared gradients, including on a language task @zhu_deep_2019.
Secure aggregation can hide individual client updates from the aggregator while it computes their
sum @bonawitz_practical_2017, but it does not protect messages that must remain visible for other
protocol stages, such as per-client split-layer activations.

Split execution also replaces raw-input transfer with an exposure surface at the cut layer. An
honest-but-curious server can use intermediate representations, gradients, and protocol knowledge to
attempt input, model, or label inference; these attacks have been demonstrated for plaintext split
learning @erdogan_unsplit_2021. Consequently, moving a cut layer or retaining an initial layer
locally should be understood as reducing direct exposure, not as a formal guarantee.

Protection requires an explicit threat model and mechanism. Secure aggregation addresses visibility
of individual aggregated updates; differential privacy adds a defined privacy accounting and
typically trades privacy against utility @abadi_deep_2016. Secure computation or trusted execution
environments can reduce what another party learns during computation, but impose their own system
assumptions and costs. These mechanisms address confidentiality risks differently from integrity
risks such as poisoning or backdoors, which require separate defenses.
