# Distributed LLMs

A structured Typst knowledge base for studying federated and distributed large language models.
It connects reusable explanations of foundational concepts and recurring research problems with
critical notes on individual papers.

## Contents

- `background/` - established concepts and broadly used techniques, including federated learning,
  LoRA, and QLoRA.
- `issues/` - research problems that recur across multiple papers, such as data heterogeneity and
  communication cost.
- `papers/` - notes on individual papers, covering their contribution, method, and assessment.
- `reference.bib` - bibliography records cited by the Typst sources.
- `paper-list.csv` - reading and coverage tracker.
- `Distributed-LLMs.typ` - root document that assembles the knowledge base.

## Build and format

Install [Typst](https://typst.app/), then run the following from the repository root:

```powershell
typst compile Distributed-LLMs.typ
```

This command produces `Distributed-LLMs.pdf`; generated PDFs are intentionally ignored by
Git.

## Adding a paper note

1. Add the source to `reference.bib` and use its BibTeX key for claims and experimental results.
2. Create `papers/<paper-name>.typ` using exactly this structure:

   ```typst
   == Paper Title <paper-short-label>

   === Summary

   === Issues Addressed

   === Method

   === Pros and Cons

   ==== Pros

   ==== Cons
   ```

3. Include the note under `= Papers` in `Distributed-LLMs.typ`.
4. Run the formatter and compile the root document.

Reuse existing background and common-issue sections through semantic Typst labels rather than
repeating their general explanations. Keep a paper's particular framing, method, evidence, and
limitations in its paper note. Shared background belongs in `background/`; add an issue section
only once substantive material is duplicated across at least two paper notes.

When contributing Markdown notes, reorganize and convert them into valid Typst rather than copying
the Markdown directly. See [AGENTS.md](AGENTS.md) for the complete editorial, citation, and
maintenance guidelines.

## Scope

Federated learning keeps raw data local, but it is not by itself a formal privacy guarantee.
Notes should distinguish data locality from protections such as secure aggregation and differential
privacy, and should keep claims tied to their sources.
