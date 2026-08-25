# Project Guidelines

## Purpose

This repository is a collection of papers and study notes about federated large language models.
The Typst document should develop into a structured knowledge base, not a sequence of isolated,
self-contained paper summaries.

## Content Organization

Organize material into these three conceptual sections:

- `Background Knowledge` explains established concepts and widely used methods needed to understand
  the field, such as federated learning, FedAvg, LoRA, and QLoRA.
- `Common Issues` explains recurring research problems addressed by multiple papers, such as data
  heterogeneity, communication cost, aggregation, privacy leakage, and malicious clients.
- `Papers` contains the contribution, method, and assessment specific to each paper.

Use the corresponding directories as the collection grows:

- `background/` for reusable background material.
- `issues/` for reusable discussions of common research issues.
- `papers/` for individual paper notes.

Do not create a separate `methods/` section or directory. Put a widely used method in `Background
Knowledge`. Keep a method introduced or substantially developed by a particular paper in that
paper's note.

## Avoiding Duplication

Before adding an explanation to a paper note, check whether the concept or issue is already covered
in `Background Knowledge` or `Common Issues`.

- Reference an existing canonical section instead of repeating its general explanation.
- Use stable, semantic Typst labels, such as `<issue-data-heterogeneity>`, `<background-fedavg>`, and
  `<paper-fedit>`.
- Keep the paper's particular framing, proposed solution, evidence, and limitations in its paper
  note.
- A common-issue section may summarize and compare broad solution families, but detailed paper
  methods belong in their respective paper notes.
- Keep shared background and common-issue sections paper-neutral. Do not explain how individual
  papers address a shared topic there; keep those paper-specific solutions in their paper notes.
- Create shared material only when it is established background or genuinely useful across papers.
  Do not create a new file merely to eliminate a small amount of repeated text.
- Cite the appropriate original sources in shared sections. A cross-reference does not replace a
  citation.

### Creating New Shared Sections

Creating or substantially expanding reusable sections in `background/` or `issues/` is a separate
task from adding an individual paper note. It requires reviewing all affected notes so that the
shared section represents the collection accurately and existing duplication can be removed
consistently.

The `Common Issues` section is duplication-driven. Keep it empty when the collection contains only
one paper note, or when an issue appears substantively in only one paper note. Do not create an
`issues/` file merely because a topic is likely to recur in future papers. Create a common-issue
section only after meaningful explanatory content about the same issue is duplicated across two or
more paper notes; until then, keep the issue discussion in the paper note where it is supported.

When adding a paper note:

- Reuse and reference canonical sections that already exist.
- Do not create a new background or common-issue section merely because the new note overlaps with
  existing notes.
- Do not refactor existing paper notes to extract newly discovered shared content unless the user
  has explicitly requested that separate task.
- Compare the new note with the existing collection and identify meaningful explanations or issues
  that are now duplicated but do not yet have a canonical section.
- After adding the paper note, report those candidates to the user and ask whether they should be
  extracted into new shared sections as a follow-up task. Briefly identify the affected notes and
  the proposed shared topic.
- If there are no meaningful new duplication candidates, say so; do not propose sections for minor
  wording overlap.

## Required Paper Note Structure

Every paper note must contain exactly these four level-three sections, in this order:

1. `Summary`
2. `Issues Addressed`
3. `Method`
4. `Pros and Cons`

Use this structure:

```typst
== Paper Title <paper-short-label>

=== Summary

=== Issues Addressed

=== Method

=== Pros and Cons

==== Pros

==== Cons
```

Do not add other level-three sections. Incorporate motivation, experiments, results, comparisons,
limitations, related work, and takeaways into the four required sections.

Lower-level headings may be used within a required section when needed for readability. `Pros and
Cons` should normally contain the level-four subsections `Pros` and `Cons`.

### Section Responsibilities

- `Summary` gives a concise overview of the research goal, central idea, and contribution. Do not
  reproduce the paper's abstract.
- `Issues Addressed` identifies the problems targeted by the paper. Reference canonical sections in
  `issues/` where they exist, and include only the paper-specific framing here.
- `Method` explains how the proposed approach works, including its algorithm, training workflow,
  architecture, aggregation strategy, or other relevant mechanics.
- `Pros and Cons` critically assesses the work. Clearly distinguish claims supported by the paper's
  evidence from the note author's own analysis.

## Adding Notes Supplied as Markdown

Users may provide paper notes in Markdown that do not follow the required paper-note structure.
Treat the Markdown as source material, not as text to copy directly into the collection. Adding such
notes requires two distinct steps:

1. Extract and reorganize the supplied content into the four required sections: `Summary`, `Issues
   Addressed`, `Method`, and `Pros and Cons`.
2. Convert the reorganized note from Markdown to valid Typst and add it as a `.typ` file under
   `papers/`.

When restructuring supplied notes:

- Preserve the technical meaning and useful detail of the source notes.
- Classify content by its purpose rather than by its original Markdown heading.
- Merge repeated passages and remove organizational duplication.
- Reference the appropriate `background/` or `issues/` content when a canonical section exists. If
  none exists, keep the supplied content in the new paper note for now and report meaningful
  collection-wide duplication as a possible follow-up task; do not create a shared section during
  paper-note ingestion.
- Place experimental findings and comparisons in the required section where they support the
  summary, method, or critical assessment; do not create additional level-three sections for them.
- Do not invent missing methods, results, advantages, limitations, citations, or factual claims. If
  the supplied material does not support a substantive claim for a required section, keep that
  section concise and state the limitation of the available notes when appropriate.
- Retain a clear distinction between statements reported by the paper and analysis expressed in the
  supplied notes.

When converting Markdown to Typst:

- Use the required Typst heading levels and the established semantic label convention.
- Convert Markdown links, emphasis, lists, code, quotations, tables, and mathematical notation into
  valid Typst syntax rather than leaving Markdown syntax in the file.
- Replace suitable repeated explanations with Typst cross-references to canonical labeled sections.
- Use BibTeX citation keys from `reference.bib`. Add or correct a bibliography entry when the
  necessary source metadata is provided or can be reliably established.
- Add the new paper file to the `Papers` includes in `Distributed-LLMs.typ`.
- Format the resulting Typst source and verify that the root document compiles successfully.

## Sources and Citations

- Do not edit `reference.bib`.
- Cite papers with their existing BibTeX keys.
- Cite a paper only once in its individual paper note, at its first substantive mention. Use
  citations in shared background or common-issue sections as needed to support their sources.
- Keep experimental numbers and claims tied to their cited sources.
- Do not present federated learning as an automatic or formal privacy guarantee. Distinguish data
  locality from protections such as secure aggregation and differential privacy.

## Typst Style and Maintenance

- The root document is `Distributed-LLMs.typ`.
- Include reusable material and paper notes from the root document rather than duplicating it there.
- Format Typst sources with `python format_typst.py`. The formatter uses Typstyle with a 100-character
  line width and wrapped prose.
- Preserve the existing heading hierarchy and IEEE bibliography style.
- Keep generated PDF files out of version control.
