# Experiment specification

**Repository URL:** `[PLACEHOLDER: INSERT GITHUB REPOSITORY URL HERE]`

This specification was written as the target behavior for the executable experiment. The code in `scripts/run_experiment.js` implements this protocol.

## 1. Micro-world

The micro-world is a science-fiction laboratory setting called **The Eos Gate**. A crew places named artifacts inside a cosmic gate. Each ignition produces one of two outcomes:

- `COHERENT`: the scene remains internally consistent.
- `FRACTURED`: the scene splits into contradictory histories.

Each artifact has a short textual card. Example:

```text
Helioforge Key: a sunlit, iron, electric, red relic, carried by gate pilots.
```

The observed semantic vocabulary is:

```text
sunlit, moonlit, iron, glass, electric, organic,
red, blue, ancient, silent, floating, engraved
```

## 2. Hidden oracle

The data generator uses one hidden semantic property:

```text
q = 2 sunlit - 2 moonlit + iron - glass + electric - organic
```

Distractor descriptors have zero contribution.

A scene is coherent iff the total scene charge is zero:

```text
COHERENT(scene) iff sum(count(object) * q(object)) = 0
```

The hidden oracle is saved in `data/hidden_oracle_for_reproducibility.json`, but the abduction functions do not receive it.

## 3. Inputs available to the abductor

The abductor receives:

1. parsed artifact cards: object name plus observed attributes;
2. training event logs: multiset of object names plus outcome label.

The parser is deterministic and scans the controlled vocabulary. No LLM API is called during the experiment. This keeps the result reproducible and prevents hidden model behavior from being confused with the abductive mechanism.

## 4. Hypotheses to compare

The experiment compares the following approaches:

1. majority label baseline;
2. exact scene memorizer;
3. 3-nearest-neighbor classifier on feature counts;
4. logistic feature-bag classifier;
5. object-ID invariant abductor;
6. HDC residual threshold using the discovered semantic concept;
7. semantic abductive invariant.

## 5. Success criteria

A run counts as successful semantic abduction only if all of the following are true:

1. the semantic abductor returns a non-trivial sparse concept over attributes;
2. the concept fits the training logs;
3. it predicts held-out known-object logs;
4. it extrapolates to larger object counts;
5. it predicts logs made from novel objects using only their descriptions;
6. it proposes counterfactual repairs that the hidden oracle verifies;
7. a shuffled-label negative control does not recover the hidden concept or does not generalize.

## 6. Failure modes that would falsify the claim

The claim would fail if:

- the semantic abductor only fit training data but failed on held-out data;
- the object-ID baseline performed equally on novel objects;
- the shuffled-label control recovered a high-performing concept;
- counterfactual repairs failed under the oracle;
- the code path gave the hidden law directly to the abductor.

## 7. Scope limitation

The test is intentionally narrow. It demonstrates abduction inside a pre-specified class of additive semantic invariants. It does not demonstrate unrestricted autonomous scientific discovery.
