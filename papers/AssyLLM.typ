== AssyLLM: Efficient Federated Fine-tuning of LLMs via Assembling Pre-trained Blocks <paper-assyllm>

=== Summary

AssyLLM@zhanAssyLLMEfficientFederated is a system for adapting large language models through
federated computation on resource-constrained edge devices. Conventional federated fine-tuning
requires each client to hold model parameters, intermediate activations, gradients, and optimizer
states during local training. AssyLLM addresses this client memory bottleneck by assembling a model
from reusable transformer blocks taken from several pretrained models, while keeping most of the
large blocks frozen.

Clients use a small amount of local data primarily to evaluate candidate blocks. The server combines
their preferences to construct an assembled model, and lightweight connection components are trained
only when selected blocks cannot be connected directly. The design therefore changes the client task
from large-scale parameter fine-tuning to block selection and model assembly. The supplied notes do
not include experimental results, so the advantages below are design rationale rather than measured
outcomes.

=== Issues Addressed

AssyLLM builds on @background-federated-learning[Federated Learning], in which raw client data stays
local while model-related information is exchanged. Its primary target is the memory bottleneck in
local LLM fine-tuning. Even parameter-efficient methods such as @background-lora[LoRA: Low-Rank
  Adaptation] still require forward computation, activation storage, backward propagation, and
training states. Activation recomputation, memory offloading, and CPU or GPU swapping can reduce
peak memory, but they trade memory savings for extra computation, storage I/O, or longer training
time.

This bottleneck is also connected to @issue-resource-heterogeneity[Resource Heterogeneity and
  Configuration Adaptation]. Devices that cannot fit the local training workload cannot contribute
their data, reducing the diversity represented in the federation and potentially worsening the
resulting model. Candidate-block selection may also reduce the size of information exchanged during
synchronization, relating the design to @issue-communication-cost[Communication Cost and
  Synchronization].

Local data is not transferred, but block preferences, activations used for comparison, or other
client messages may still reveal information. AssyLLM therefore does not provide a formal privacy
guarantee, and the relevant concern remains @issue-privacy-leakage[Privacy Leakage Beyond Data
  Locality]. Heterogeneous client data can also affect which blocks are preferred, making
@issue-data-heterogeneity[Data Heterogeneity and Client Drift] relevant to the selection and
aggregation process.

=== Method

==== Block Pool

The server starts with several existing pretrained models and divides each into groups of
transformer layers called blocks. These blocks form a shared block pool. Rather than modifying every
parameter of one model, the system searches for useful combinations of components from the pool. A
completed model may contain an early block from one source model, a middle block from another, and a
later block from a third.

==== Client-Side Block Selection

When a partial assembled model needs its next block, each client evaluates the available candidates.
For each candidate, the client attaches it to the current model, runs forward inference on a small
amount of local data, collects activation information, calculates compatibility, and ranks the
candidate. Clients send their block preferences to the server. The server aggregates those
preferences and selects the next block, repeating the process until the assembled model is complete.
Client work is therefore mainly forward computation and evaluation rather than full backpropagation
through a large model.

==== Block Comparator

The Block Comparator combines two complementary measurements, centered kernel alignment (CKA) and
layer correlation (COR). CKA measures similarity between learned representations. For a candidate
block, the system compares the representation after insertion with the representation produced in
the block's original pretrained model. Greater similarity suggests that the candidate may be more
compatible.

COR considers intermediate layer behavior as well as the resulting representation. It compares
activation distributions between corresponding layers and measures their differences. CKA thus
focuses on representation similarity, while COR also examines the similarity of the internal path.
AssyLLM combines both measurements when ranking candidate blocks. They remain proxy measures, rather
than guarantees of end-task quality.

==== Elastic Adapter

Blocks from different models may have different hidden dimensions, feature representations,
attention structures, or other internal architectural properties. The Elastic Adapter connects such
blocks when they cannot be composed directly. A projection can map an output representation to the
next block's input dimension, such as mapping 4096 dimensions to 1024 dimensions. When matching
dimensions do not ensure matching semantics, more involved alignment mechanisms, such as
cross-attention, can be used.

The large pretrained blocks remain frozen. Only the relatively lightweight connection components
need adjustment when an adapter is required.

==== Block Quanter

The block pool itself can require substantial storage and memory when it contains blocks from
several large models. Block Quanter applies mixed numerical precision according to weight
importance. It estimates how strongly individual weights affect a block's output, then assigns
higher precision to more important weights and lower precision to less important weights. The
importance analysis can be performed offline. This preserves more information where it matters while
compressing the block pool more aggressively elsewhere.

==== Block Swapper

Quantization may still leave the block pool too large for device memory. Block Swapper keeps
currently useful blocks in fast memory and leaves other blocks in slower storage. It considers both
recent use and relevance to the current assembled model when deciding which blocks to evict. Likely
future blocks can be preloaded, overlapping storage I/O with computation, and blocks can be removed
before memory is full to reduce pauses while waiting for space.

==== Complete Workflow

The complete workflow is:

1. Collect several pretrained models and split them into reusable blocks.
2. Quantize the blocks and place them in the shared block pool.
3. Begin constructing a new model from the available blocks.
4. Have clients evaluate candidate blocks with local data and forward inference.
5. Use CKA and COR to estimate candidate compatibility.
6. Aggregate client preferences and select the next block.
7. Insert an Elastic Adapter when the selected blocks need a connection component.
8. Swap blocks between storage and memory as needed.
9. Repeat selection and assembly until the model is complete.

=== Pros and Cons

==== Pros

- The design avoids requiring clients to fully train a large LLM, reducing the need for full
  gradient storage, large optimizer states, and backward activations.
- Lower client-side requirements may allow more resource-constrained devices to participate,
  bringing a broader range of distributed private data into the federation.
- Reusing blocks treats pretrained models as repositories of existing knowledge instead of requiring
  the client process to learn all task behavior again.
- Blocks from multiple source models can potentially combine useful early, middle, and late
  representations in one assembled model.
- Communicating block-selection information and lightweight trainable components can be smaller than
  communicating complete updates for a very large model, addressing part of the communication
  burden.
- The modular components separate block compatibility, cross-block connection, block-pool
  compression, and memory management, making the system's responsibilities easier to reason about.

==== Cons

- The assembled model depends on the quality and coverage of the block pool. It cannot create task
  knowledge that is absent from the source models.
- Adding source models increases the number of blocks and creates storage, quantization, and
  swapping overhead. The approach shifts part of the resource problem from training memory to
  block-pool management.
- Independently trained blocks may differ in dimensionality, representation, architecture, attention
  mechanism, or token-level semantics. The comparator and adapter reduce these problems but cannot
  guarantee that every apparently compatible combination will behave well.
- Keeping large blocks frozen constrains the adaptation space compared with full parameter
  fine-tuning. Tasks requiring changes poorly represented by the source models may benefit from the
  greater capacity of conventional fine-tuning.
- CKA and COR are compatibility approximations. Similar representations or internal behavior do not
  guarantee the best end-task model.
- The system introduces management and coordination complexity through block selection,
  compatibility calculation, adapters, mixed-precision storage, swapping, preloading, and
  server-side preference aggregation.
- The supplied notes contain no experimental results, so these design benefits and limitations
  cannot be assessed here with reported measurements.
