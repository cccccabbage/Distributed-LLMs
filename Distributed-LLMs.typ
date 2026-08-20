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

= Common Issues

#include "issues/data-heterogeneity.typ"

= Papers

#include "papers/FedIT.typ"
#include "papers/Dec-LoRA.typ"

= References

#bibliography("reference.bib", style: "ieee", title: none)
