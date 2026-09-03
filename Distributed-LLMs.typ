#set document(title: "Distributed LLMs")
#set page(margin: 2cm)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 22pt, weight: "bold")
#show heading.where(level: 2): set text(size: 18pt, weight: "bold")
#show heading.where(level: 3): set text(size: 14pt, weight: "bold")
#show ref: set text(fill: rgb("#0b5f8a"))

#align(center)[#text(size: 28pt, weight: "bold")[Distributed LLMs]]

#outline(
  title: [Contents],
  depth: 2,
)

= Background Knowledge

#include "background/federated-learning.typ"
#include "background/LoRA.typ"
#include "background/QLoRA.typ"
#include "background/prefill-decode-kv-cache.typ"
#include "background/reinforcement-learning.typ"

= Common Issues

#include "issues/data-heterogeneity.typ"
#include "issues/communication-cost.typ"
#include "issues/resource-heterogeneity.typ"
#include "issues/inference-resource-heterogeneity.typ"
#include "issues/privacy-leakage.typ"
#include "issues/untrusted-permissionless-compute.typ"
#include "issues/asynchronous-update-staleness.typ"

= Papers

#include "papers/FedIT.typ"
#include "papers/Dec-LoRA.typ"
#include "papers/FedSpine.typ"
#include "papers/DiLoCo.typ"
#include "papers/HexGen.typ"
#include "papers/HexGen-2.typ"
#include "papers/Eager-Updates.typ"
#include "papers/Helix.typ"
#include "papers/HLoRA.typ"
#include "papers/Rank-Heterogeneity.typ"
#include "papers/FAH-QLoRA.typ"
#include "papers/Preble.typ"
#include "papers/EdgeShard.typ"
#include "papers/SflLLM.typ"
#include "papers/Gauntlet.typ"
#include "papers/INTELLECT-2.typ"
#include "papers/Jupiter.typ"
#include "papers/MDI-LLM.typ"
#include "papers/Photon.typ"
#include "papers/Seesaw.typ"

= References

#bibliography("reference.bib", style: "ieee", title: none)
