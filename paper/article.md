# The Eos Gate: A Minimal Semantic Micro-World for Testing Conceptual Abduction with Vector-Symbolic Representations

**Repository placeholder:** `[PLACEHOLDER: INSERT GITHUB REPOSITORY URL HERE]`

## Abstract

This paper describes a deliberately small but falsifiable experiment for testing a narrow form of conceptual abduction. The setting is a science-fiction micro-world, **The Eos Gate**, where imaginary artifacts are described by short textual cards and gate-ignition logs are labeled either **COHERENT** or **FRACTURED**. The hidden generator uses a sparse semantic invariant, but the abductor receives only parsed artifact descriptions and labeled event logs. The system must introduce a latent semantic property that makes the story-world coherent. In the included run, the semantic abductor recovers the hidden property exactly, obtains 1.000 accuracy on training, held-out known-object events, larger-count extrapolation events, and events made only from novel objects described by words. It also proposes 30/30 counterfactual repairs that are verified by the hidden oracle. Several baselines are tested: majority prediction, exact memorization, nearest neighbors, a logistic feature-bag classifier, an object-ID invariant, and a finite-dimensional HDC residual threshold. The result is not presented as unrestricted autonomous science. It is a controlled demonstration that, inside a pre-specified additive-invariant hypothesis class, a system can introduce a compact latent concept that explains, predicts, generalizes, and intervenes.

**Keywords:** abduction; vector symbolic architectures; hyperdimensional computing; conceptual discovery; minimum description length; semantic micro-worlds; neuro-symbolic AI.

## 1. Motivation

Many AI demonstrations are hard to interpret epistemically. A classifier can achieve high accuracy without giving a reusable explanation; a language model can produce an attractive hypothesis without producing an operational object; and a symbolic solver can recover a hidden equation while leaving the user unconvinced that anything like conceptual discovery occurred. The goal of this experiment is to avoid all three ambiguities by constructing a micro-world where the abductive claim has to be inspectable.

The target operation is not general scientific creativity. It is smaller and more precise: given textual descriptions of objects and observations of their behavior, can a system introduce a new latent property that was not explicitly named in the data, use it to compress the observations, and then apply it to unseen cases? In philosophical terms, abduction is often characterized as inference to the best explanation. In computational terms, we operationalize that idea with a minimum-description-length style criterion: the preferred hypothesis should reduce the combined cost of the model plus the residual data errors. A mere label predictor is not sufficient. The system must return a concept that can be written down, audited, and used to make counterfactual repairs.

The Eos Gate universe is intentionally accessible. A crew places artifacts into a cosmic gate. Some configurations remain coherent; others fracture the timeline. Each artifact is described with ordinary words such as `sunlit`, `moonlit`, `iron`, `glass`, `electric`, and `organic`. The hidden rule is a semantic conservation law, but the system is not given the law. It sees cards and logs. If it discovers that coherence depends on a neutral balance between positive and negative semantic contributions, it has introduced an explanatory latent property.

## 2. What counts as abduction here?

The word *abduction* can easily become hand-waving. Therefore the experiment uses a strict operational criterion. A run counts as a successful semantic abduction only if the system:

1. receives artifact descriptions and event labels, not the hidden oracle;
2. introduces a non-trivial latent property over semantic attributes;
3. explains training observations with that property;
4. predicts held-out events made from known objects;
5. extrapolates to larger object counts;
6. predicts events made from novel objects using only their descriptions;
7. proposes counterfactual repairs that the hidden oracle verifies; and
8. fails, or at least generalizes poorly, when the training labels are shuffled.

This definition distinguishes the experiment from ordinary classification. A classifier learns a function from feature vectors to labels. The abductive model instead proposes an intermediate entity: a sparse semantic quantity `q`. Once `q` exists, it explains why the same object can participate in multiple coherent or fractured scenes, why some new objects should behave like old ones, and how to repair a fractured scene.

The criterion also distinguishes the experiment from pure memorization. An object-ID invariant can learn charges for named objects seen during training. That is useful, but it cannot assign a charge to a new object from its textual description. The semantic abductor must do exactly that. Novel-object evaluation is therefore central rather than decorative.

## 3. Micro-world specification

The observed vocabulary contains twelve descriptors:

`sunlit, moonlit, iron, glass, electric, organic, red, blue, ancient, silent, floating, engraved`.

The data generator, and only the data generator, uses the following hidden property:

```text
q = 2 sunlit - 2 moonlit + iron - glass + electric - organic
```

The remaining descriptors are distractors with zero contribution. A few example cards are:

```text
Helioforge Key: a sunlit, iron, electric, red relic, carried by gate pilots.
Nulltide Lantern: a moonlit, glass, organic, blue lamp, grown in quiet oceans.
Spark Coin: an iron, electric, red coin, used as a battery-token.
Glass Moth: a glass, organic, floating moth, alive only in vacuum gardens.
```

An event is a multiset of artifacts. The hidden oracle assigns each scene a total charge and labels the scene as coherent exactly when the total is zero. Formally, let `a(o) in {0,1}^m` be the attribute vector for object `o`, let `c_o` be the count of that object in a scene, and let `q in Z^m` be the latent semantic property. The scene vector is:

```text
x(scene) = sum_o c_o a(o)
```

and the oracle rule is:

```text
COHERENT(scene) iff q^T x(scene) = 0.
```

The important point is that the abductor is not told this `q`. It receives the cards, parses the observed attributes, and receives event labels. The hidden law is stored in the repository only for reproducibility and final verification.

## 4. Method

### 4.1 Parsing and representation

The runtime experiment does not call an LLM. It uses a deterministic parser that scans the controlled vocabulary in each artifact card and extracts object counts from event logs. This design choice is deliberate: the claim should depend on the abductive mechanism, not on undocumented behavior of a remote language model. In a larger LLM+VSA system, the parser could be replaced by an LLM, while the abductive test would remain the same.

For the semantic abductor, each event becomes a feature-count vector `x`. For the HDC/VSA representation, each attribute receives a random bipolar hypervector `r_f in {-1,+1}^D`. An artifact vector is a bundle of its attribute vectors:

```text
v(o) = sum_{f in attrs(o)} r_f.
```

A candidate concept is encoded as:

```text
C(q) = sum_f q_f r_f.
```

The HDC residual for a scene is:

```text
r_D(scene, q) = (1/D) < sum_o c_o v(o), C(q) >.
```

Because random hypervectors are approximately orthogonal in high dimension, `r_D(scene,q)` approximates `q^T x(scene)`. In finite dimension, cross-talk remains. That is why the paper reports the HDC residual threshold separately from the decoded exact semantic invariant.

### 4.2 Semantic invariant abduction

The abductor uses the rows labeled `COHERENT` and asks whether they admit a non-zero vector `q` such that:

```text
X_+ q = 0,
```

where `X_+` is the matrix of coherent event feature-count vectors. The implementation computes the nullspace with a dependency-free reduced-row-echelon-form routine. If a non-trivial nullspace exists, it decodes a small integer vector and validates it against both coherent and fractured training examples. This is the abductive step: the system introduces an unobserved latent property that makes a class of observations simple.

The selection principle is MDL-like. A useful hypothesis is not merely one that fits labels; it should be compact. The reported diagnostic approximates:

```text
score(h) = L(h) + L(D | h),
```

where `L(h)` penalizes the number of active semantic terms and coefficient size, and `L(D|h)` penalizes residual mistakes. This is not claimed as a full MDL code. It is a transparent sanity check that the semantic invariant is a compact explanation, not a giant lookup table.

### 4.3 Baselines and controls

The experiment compares seven approaches.

- **Majority label** predicts the training majority class.
- **Exact scene memorizer** stores exact training multisets and defaults to the majority label otherwise.
- **3-nearest neighbors** predicts by Euclidean distance between semantic feature-count vectors.
- **Logistic feature-bag classifier** is a discriminative linear classifier over feature counts.
- **Object-ID invariant** applies the same nullspace idea over object names instead of semantic attributes. It can generalize to new mixtures of known objects but should fail on novel objects.
- **HDC residual threshold** uses the recovered semantic concept as a VSA/HDC concept vector and classifies by a learned residual threshold.
- **Semantic abductive invariant** is the decoded sparse semantic property and exact zero-residual rule.

The negative control shuffles training labels while preserving artifact cards and scenes. If the abductor still recovered the hidden law, the result would be suspect. In the included run, the shuffled coherent rows span the full semantic space and leave no non-trivial invariant.

## 5. Results

The semantic abductor recovered the hidden concept exactly, up to the irrelevant sign ambiguity of a zero-residual law:

| Descriptor | Hidden weight | Discovered weight |
|---|---:|---:|
| sunlit | 2 | 2 |
| moonlit | -2 | -2 |
| iron | 1 | 1 |
| glass | -1 | -1 |
| electric | 1 | 1 |
| organic | -1 | -1 |
| red | 0 | 0 |
| blue | 0 | 0 |
| ancient | 0 | 0 |
| silent | 0 | 0 |
| floating | 0 | 0 |
| engraved | 0 | 0 |


The central accuracy results are:

| Method | Train | Known test | Extrapolation | Novel objects |
|---|---:|---:|---:|---:|
| Semantic abductive invariant | 1.000 | 1.000 | 1.000 | 1.000 |
| Object-ID invariant | 1.000 | 1.000 | 1.000 | 0.500 |
| HDC residual threshold | 1.000 | 0.970 | 0.820 | 0.953 |
| Exact scene memorizer | 1.000 | 0.503 | 0.500 | 0.500 |
| 3-nearest neighbors | 0.875 | 0.680 | 0.673 | 0.657 |
| Logistic feature-bag classifier | 0.556 | 0.533 | 0.577 | 0.503 |
| Majority label | 0.500 | 0.500 | 0.500 | 0.500 |

The object-ID invariant is an especially useful comparison. It reaches perfect accuracy on known objects because the world is governed by a charge-like rule and the known-object charges can be inferred. However, it collapses to 0.500 on novel objects because it has no way to map new textual descriptions to object charges. The semantic invariant keeps 1.000 accuracy because it has inferred the underlying word-level property.

The HDC residual threshold also matters. It realizes the concept in a high-dimensional distributed vector space and obtains 1.000 on training, 0.970 on known held-out events, 0.820 on larger-count extrapolation, and 0.953 on novel objects. The extrapolation drop is not hidden. It is the expected finite-dimensional cross-talk of the distributed approximation. The decoded semantic invariant, once recovered, is the exact operational concept.

The counterfactual test is the strongest behavioral evidence. For a fractured scene, the discovered concept computes a nonzero residual and searches for additions that bring the residual to zero. The hidden oracle then verifies the repaired scene. In the included run, 30 out of 30 proposed repairs were verified. This is qualitatively different from label prediction: the system can use the concept to change a scene.

The approximate MDL-style diagnostic also favors the semantic invariant. The semantic invariant has zero training mistakes with a compact sparse model. Exact memorization also has zero training mistakes, but its model cost is much larger and it fails on held-out events. This supports the interpretation that the discovered property is an explanatory compression rather than a lookup table.

## 6. Why this is not cheating

The experiment is not cryptographic; the hidden law is present in the repository because the experiment must be reproducible. The anti-cheating claim is instead methodological and auditable.

First, the specification states the success and failure criteria before the implementation. The relevant document is `docs/experiment_spec.md`. The code implements that protocol and saves the generated artifacts, logs, metrics, and verification certificate.

Second, the hidden oracle is used by `src/world.js` to generate labels and by the driver to verify held-out predictions and repairs. The abduction module receives parsed attributes and labels. The reviewer can inspect `src/abductor.js`: it computes a nullspace from observed coherent rows; it does not import the hidden law.

Third, the experiment includes novel objects. A method that only memorizes object names cannot infer how a never-seen `Daysteel Compass` or `Dreamglass Seed` should behave. The semantic invariant can do so because it assigns meaning to descriptors, not just to IDs.

Fourth, the object-ID invariant baseline is intentionally strong. It shows that invariant discovery over names is not enough. A known-object invariant is real but not semantic. Its failure on novel objects is the control that makes the semantic result meaningful.

Fifth, the shuffled-label control destroys the relation between descriptions and outcomes. In the included run, the same abductor finds no non-trivial nullspace after label shuffling. If the method were merely forcing a pretty explanation onto arbitrary labels, this control would be much more likely to succeed.

Sixth, no LLM is called at runtime. Coding-assistant tools were used to create and document the repository, but the experiment itself is deterministic Node.js with no npm dependencies. This separates the scientific claim from any opaque model behavior.

## 7. Discussion and limitations

The result should be interpreted narrowly. The hypothesis class is manually chosen: sparse additive semantic invariants. The system does not discover from scratch that the universe should be governed by an additive conservation law. It discovers the concept inside that class. This limitation is important, but it does not make the test trivial. Within the class, the system must identify which words matter, which are distractors, how they contribute, whether the property generalizes to unseen objects, and whether the property supports intervention.

The science-fiction setting is not a toy only for entertainment. It solves an epistemic problem. Real scientific domains bring many confounds: measurement noise, ambiguous terminology, incomplete mechanisms, and long causal chains. The Eos Gate keeps the world readable while preserving the key structure of a scientific explanation: a latent quantity makes many observations simple.

The VSA/HDC component should also be read carefully. The present experiment uses HDC as a distributed realization of semantic concepts, not as a magical replacement for all symbolic reasoning. The HDC residual approximates the exact semantic residual and degrades under larger counts. That reported degradation is useful: it tells us where the distributed representation introduces error and where exact decoded concepts remain necessary.

A natural next step is to make the parser genuinely linguistic, to add noise, and to allow richer hypothesis classes such as relations, roles, temporal order, and non-additive interactions. Another extension is active experimentation: instead of passively receiving logs, the agent would choose the next scene that maximally distinguishes competing hypotheses.

## 8. Conclusion

The Eos Gate experiment demonstrates a minimal, inspectable form of conceptual abduction. The system introduces a latent semantic property that is absent from the labels, recovers a compact law, predicts held-out and novel-object events, and proposes verified repairs. The result is not a claim of general autonomous science. It is a controlled test for one crucial operation that an autonomous researcher would need: turning observations and words into a new operational concept that makes a world simpler.

## References

1. Douven, I. "Abduction." *The Stanford Encyclopedia of Philosophy*. The entry characterizes abduction as inference to the best explanation and discusses its contested normative status. https://plato.stanford.edu/entries/abduction/
2. Kleyko, D., Rachkovskij, D. A., Osipov, E., and Rahimi, A. "A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations." arXiv:2111.06077, 2021.
3. Kleyko, D., Rachkovskij, D. A., Osipov, E., and Rahimi, A. "A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges." arXiv:2112.15424, 2021.
4. Grünwald, P. *The Minimum Description Length Principle*. MIT Press, 2007. See also tutorial introductions to MDL as model selection by compression.
5. Barron, A., Rissanen, J., and Yu, B. "The Minimum Description Length Principle in Coding and Modeling." *IEEE Transactions on Information Theory*, 44(6), 1998.
6. Sun, Z.-H., Zhang, R.-Y., Zhen, Z., Wang, D.-H., Li, Y.-J., Wan, X., and You, H. "Systematic Abductive Reasoning via Diverse Relation Representations in Vector-symbolic Architecture." arXiv:2501.11896, 2025.
