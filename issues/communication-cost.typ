== Communication Cost and Synchronization <issue-communication-cost>

Collaborative model training repeatedly exchanges model parameters, gradients, or updates. The
communication burden has two separate dimensions: the number of synchronization rounds and the size
of each exchanged state. Frequent synchronization can require low-latency, high-bandwidth links,
while a large model or update can make each exchange expensive even if rounds are sparse
@mcmahan_communication-efficient_2023 @douillard_diloco_2024.

Methods can reduce the transmitted state, the synchronization frequency, or both.
Parameter-efficient fine-tuning communicates adapters instead of the entire base model, as in FedIT
and Dec-LoRA @zhang_towards_2024 @ghiasvand_decentralized_2025. DiLoCo instead performs many local
optimizer steps before combining full-model displacements, reducing synchronization frequency but
retaining model-sized exchanges @douillard_diloco_2024.

These choices introduce trade-offs. More local work lowers the number of communication rounds but
can increase disagreement between participants; smaller or sparse communication topologies reduce
per-round traffic but may spread information more slowly. The resulting optimization consequences
are discussed in @issue-data-heterogeneity[Data Heterogeneity and Client Drift]. Communication
efficiency also does not imply lower total compute, nor does data locality alone establish privacy.
