# Methodology

## Data generation

The world generator creates balanced splits of `COHERENT` and `FRACTURED` event logs. The training and known-test splits use known artifacts. The extrapolation split uses the same known artifacts but larger counts. The novel-object split uses objects that never appear in training but have cards built from the same semantic vocabulary.

Balanced labels prevent the majority baseline from looking good by accident.

## Parsing

The experiment uses a deterministic parser because the focus is not natural-language extraction. The parser identifies controlled vocabulary terms in the artifact cards and recovers object counts from event logs.

In a larger LLM+VSA architecture, this parser would be replaced by an LLM. Here it is deliberately simple so that the abductive claim depends on the invariant search, not on hidden LLM behavior.

## Semantic abductor

For each event, the code constructs a feature-count vector `x`. The semantic abductor uses the `COHERENT` rows and asks whether there is a non-zero vector `q` such that:

```text
x dot q = 0
```

It computes the nullspace of the stable-event matrix with a dependency-free RREF implementation. If a nullspace exists, it decodes a small integer vector and validates it against both stable and fractured examples.

This is an abductive move because the system introduces an unobserved property `q` to explain why the observed events divide into coherent and fractured cases.

## Object-ID invariant baseline

The object-ID invariant uses the same nullspace idea, but over object names rather than semantic attributes. It can recover charges for known objects, but it has no way to assign a charge to a novel object from a description. This baseline is important because it separates ordinary object memorization from semantic concept formation.

## VSA/HDC representation

The HDC module assigns random bipolar hypervectors to semantic attributes. Artifact vectors are bundles of attribute hypervectors, and concept vectors are weighted bundles of attribute hypervectors. The HDC residual approximates the semantic dot product. Because the representation is finite-dimensional, cross-talk can reduce extrapolation accuracy. This is reported rather than hidden.

## Negative control

The training labels are shuffled while the scenes and artifact cards are kept fixed. The same semantic abductor is run on this corrupted evidence. In the included run, the shuffled stable rows span the full semantic space, leaving no non-trivial nullspace. That failure is evidence that the original result depends on real structure in the labels.

## Counterfactual repairs

A fractured scene has nonzero residual. The system searches for a small addition of known or novel artifacts that makes the residual zero. The repaired scene is then checked with the hidden oracle. This tests whether the concept is generative and intervention-capable, not merely descriptive.
