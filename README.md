# The Eos Gate: a semantic abduction micro-world

**GitHub repository URL:** `[PLACEHOLDER: INSERT GITHUB REPOSITORY URL HERE]`

This repository contains a dependency-free Node.js experiment for testing whether a small neuro-symbolic system can perform a minimal form of **conceptual abduction** in a readable science-fiction micro-world.

The world contains short textual cards for imaginary artifacts and event logs from the **Eos Gate**. The gate either remains `COHERENT` or becomes `FRACTURED`. The hidden generator uses a sparse latent semantic property, but the abductor is not given that property. It receives only artifact descriptions, parsed semantic attributes, and labeled event logs.

The main claim is intentionally narrow:

> Inside a pre-specified additive-invariant hypothesis class, the system can introduce a latent semantic property that explains the logs, generalizes to novel objects described only by words, and proposes counterfactual repairs verified by the hidden oracle.

It does **not** claim unconstrained autonomous scientific discovery.

## Quick start

Requirements:

- Node.js 18+ or newer.
- No npm dependencies.

Run everything:

```bash
npm run all
```

Equivalent make command:

```bash
make all
```

This regenerates:

- `data/*.json` and `data/*.md`: artifact cards and event logs.
- `results/summary.json`: full machine-readable result.
- `results/metrics.csv`: accuracy table.
- `results/hypothesis_comparison.csv`: MDL-style comparison.
- `results/abduction_certificate.md`: human-readable verification certificate.
- `results/interventions.csv`: counterfactual repairs.
- `figures/*.svg`: generated figures.

Run only the experiment:

```bash
node scripts/run_experiment.js
```

Run verification checks:

```bash
npm test
```

The test script prints a detailed report rather than a single `OK` line.

## Repository map

```text
src/
  world.js              # micro-world, artifact cards, hidden oracle for data generation
  parser.js             # deterministic parser for synthetic cards/logs
  linear_algebra.js     # dependency-free RREF/nullspace utilities
  features.js           # semantic and object-count feature construction
  hdc.js                # dependency-free VSA/HDC vectors and residuals
  abductor.js           # semantic and object-ID invariant abductors
  baselines.js          # majority, memorizer, kNN, logistic classifier
scripts/
  run_experiment.js     # main experiment driver
  generate_figures.js   # dependency-free SVG figure generator
test/
  verify_results.js     # checks claims against saved results
docs/
  experiment_spec.md
  methodology.md
  human_review_and_agent_use.md
paper/
  eos_gate_semantic_abduction.docx
  article.md
results/
  summary.json
  metrics.csv
  abduction_certificate.md
  interventions.csv
figures/
  pipeline.svg
  concept_weights.svg
  accuracy_bars.svg
  hypothesis_mdl.svg
  interventions.svg
```

## Why the experiment is not just a classifier demo

A classifier learns `input -> label`. The semantic abductor instead searches for a sparse latent property `q` over words and then tests the law:

```text
COHERENT iff total q-charge of the scene is zero.
```

That property is then used to:

1. explain individual events;
2. predict held-out events;
3. extrapolate to larger object counts;
4. predict scenes made only from novel objects; and
5. propose repairs to fractured scenes.

The object-ID invariant baseline can learn known-object charges and generalize to known-object mixtures, but it fails on novel objects because it has no semantic bridge from descriptions to charges. This is why the semantic invariant is the stronger abductive result.

## Main result from the included run

The hidden oracle was:

```text
sunlit   +2
moonlit  -2
iron     +1
glass    -1
electric +1
organic  -1
red       0
blue      0
ancient   0
silent    0
floating  0
engraved  0
```

The semantic abductor recovered exactly the same concept up to sign.

Accuracy summary:

| Method | Train | Known test | Extrapolation | Novel objects |
|---|---:|---:|---:|---:|
| Semantic abductive invariant | 1.000 | 1.000 | 1.000 | 1.000 |
| Object-ID invariant | 1.000 | 1.000 | 1.000 | 0.500 |
| HDC residual threshold | 1.000 | 0.970 | 0.820 | 0.953 |
| Exact scene memorizer | 1.000 | 0.503 | 0.500 | 0.500 |
| 3-nearest neighbors | 0.875 | 0.680 | 0.673 | 0.657 |
| Logistic feature-bag classifier | 0.556 | 0.533 | 0.577 | 0.503 |
| Majority label | 0.500 | 0.500 | 0.500 | 0.500 |

Negative control: when training labels are shuffled, the semantic abductor finds no non-trivial nullspace in the rows labeled `COHERENT` and therefore fails to recover the hidden concept.

## Methodological caveat

The hypothesis space is pre-specified: sparse additive semantic invariants. That is intentional. The goal is not to demonstrate general discovery from arbitrary worlds, but to create a falsifiable, understandable micro-test for the specific abductive operation: introducing a latent semantic quantity that makes a story-world coherent.
