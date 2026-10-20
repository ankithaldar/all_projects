# Module Doc Template — Cornell Birdcall 1st Place Solution

> You are producing ONE module chapter of a larger Notion-style technical
> document. Your chapter will be concatenated with others, so follow the
> headings **exactly** and in this order. Do not write a title of your own —
> the filename is the title. Do not write a table of contents. Do not
> summarise the whole system; document only your module, but document it
> exhaustively.

## Audience

The original author, in the future. They know PyTorch well. They do **not**
remember what they did. They want to be able to re-run, re-tune, and
re-derive every decision. Favour precise, checkable statements over prose
comfort.

## Hard rules

1. **Every claim must be traceable to the code you read.** Quote the exact
   identifier, default value, tensor dimension, or line. If something is
   ambiguous in the code, say so explicitly rather than guessing.
2. **Explain the WHY.** The author had a reason. Where the reason is
   documented in code comments or the README, cite it. Where it is not,
   infer it and label the inference: *(inferred — reason: …)*.
3. **Document dead/experimental code too.** This repo carries the whole
   research history. Say plainly which paths the final solution uses and
   which are abandoned. This is as valuable as the live path.
4. **Mermaid diagrams**: use ```mermaid fences. Label every edge. No
   parentheses or brackets inside node labels — they break the parser. Use
   `flowchart TD` and quote any label containing spaces.
5. **Tensor shapes**: always write them as `(B, C, T)` / `(B, F, T, M)`.
   State B, C, T, F, M concretely using the real config values.
6. **Code blocks**: only for illustrating a point that prose cannot carry
   (a formula, a shape dance, a threshold rule, a non-obvious trick). Never
   paste a file wholesale.
7. Use Notion flavour: `> [!info]` / `> [!warning]` / `> [!tip]` callouts,
   `**bold**` for key concepts, `-` bullets, `###` sub-headings.
8. Length: go long. Exhaustive beats short. Do not pad with filler.

## Required heading skeleton

```
## Purpose

<2–5 sentences: what this module is responsible for, in the overall
pipeline, and why it exists as a separate unit.>

## Contracts

**Inputs** — exact tensors/objects and their shapes and dtypes.
**Outputs** — exact return values and shapes.
**Called by** — which module calls it, with what.
**Calls** — which modules it depends on.

## How it works

<step-by-step, numbered where the order matters. Name every function.>

## Key decisions and tricks

<the non-obvious parts. This is the most important section.>

## Failure modes and gotchas

<what breaks, and what the code does about it.>

## Mermaid

<one or more diagrams>
```

## Return value

When finished, reply with a compact bullet list (max 25 lines) of:
- the 5 most important facts in your module,
- any discrepancy or bug you found between code and README,
- anything you could NOT determine from the code.
