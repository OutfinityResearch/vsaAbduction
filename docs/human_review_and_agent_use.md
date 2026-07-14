# Coding-agent assistance, interventions, and human review

This repository is designed to be transparent about how the experiment was produced.

## Agent assistance

Coding-assistant tools were used to draft and refactor the experiment, generate documentation, and assemble the editable article. This is not presented as an autonomous scientific discovery by the coding agent. The scientific object is the executable protocol in the repository.

## Human review expectation

A human reviewer should inspect:

1. `docs/experiment_spec.md` for the pre-specified success and failure criteria;
2. `src/world.js` to verify that the hidden oracle is isolated in the generator;
3. `src/abductor.js` to verify that the abductor receives observations, not the oracle;
4. `results/summary.json` and `results/abduction_certificate.md` to check the reported results;
5. `test/verify_results.js` to inspect what is actually being asserted.

## Human interventions during development

The experiment went through conceptual iterations. Earlier versions used anonymous symbols and were less persuasive as abduction tests. The present version was selected because it links words, objects, hidden physical coherence, held-out prediction, novel-object generalization, and counterfactual repair.

## Why disclosure matters

The point of the repository is not to hide automation but to make the result reproducible. Agent-generated code can be wrong. Therefore the repository includes a specification, deterministic code, generated data, saved results, explicit controls, and verification scripts.
