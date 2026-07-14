#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { mulberry32, shuffle } = require('../src/rng');
const {
  FEATURES,
  HIDDEN_LAW,
  KNOWN_ARTIFACTS,
  NOVEL_ARTIFACTS,
  artifactCharge,
  sceneCharge,
  sceneSignature,
  artifactCard,
  makeArtifactsByName,
  generateDataset
} = require('../src/world');
const { parseArtifactCard } = require('../src/parser');
const { sceneFeatureCounts, objectCountVector, scoreVector, vectorToObject } = require('../src/features');
const { abductSemanticInvariant, abductObjectInvariant, predictSemantic, predictObject } = require('../src/abductor');
const { majorityTrain, exactSceneMemorizer, nearestNeighbor, logisticFeatureClassifier, evaluateModel } = require('../src/baselines');
const { makeHypervectors, encodeConcept, fitThreshold, evaluateHDC, buildArtifactVectors, hdcResidual } = require('../src/hdc');

const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, 'data');
const RESULTS_DIR = path.join(ROOT, 'results');
const SEED = 424242;
const HDC_DIM = 8192;

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function writeJSON(file, obj) { fs.writeFileSync(file, JSON.stringify(obj, null, 2)); }
function writeText(file, text) { fs.writeFileSync(file, text); }
function boolLabel(x) { return x ? 'COHERENT' : 'FRACTURED'; }
function accuracyFromPredict(events, predFn) {
  let correct = 0;
  for (const e of events) if (predFn(e) === e.coherent) correct++;
  return correct / events.length;
}
function mistakesFromPredict(events, predFn) {
  let mistakes = 0;
  for (const e of events) if (predFn(e) !== e.coherent) mistakes++;
  return mistakes;
}
function log2(x) { return Math.log(x) / Math.log(2); }
function comb(n, k) {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let c = 1;
  for (let i = 1; i <= k; i++) c = c * (n - k + i) / i;
  return c;
}
function approxMdlBits(modelName, trainEvents, predictionMistakes, modelDetails) {
  const n = trainEvents.length;
  let modelBits = 8;
  if (modelName === 'Semantic abductive invariant') {
    const k = modelDetails.nonzero;
    modelBits = 12 + log2(comb(FEATURES.length, k)) + k * log2(4) + 4; // chosen features, coefficients, equality rule
  } else if (modelName === 'Object-id invariant') {
    const k = modelDetails.nonzero;
    modelBits = 12 + log2(comb(KNOWN_ARTIFACTS.length, k)) + k * log2(16) + 4;
  } else if (modelName === 'Exact scene memorizer') {
    modelBits = modelDetails.uniqueScenes * 12 + modelDetails.uniqueScenes;
  } else if (modelName.includes('Logistic')) {
    modelBits = (FEATURES.length + 1) * 16;
  } else if (modelName.includes('neighbors')) {
    modelBits = n * FEATURES.length * 4;
  } else if (modelName === 'Majority label') {
    modelBits = 1;
  } else if (modelName.includes('HDC')) {
    modelBits = 12 + log2(comb(FEATURES.length, modelDetails.nonzero)) + modelDetails.nonzero * log2(4) + 16;
  }
  const dataBits = predictionMistakes * log2(n + 1);
  return { modelBits, dataBits, totalBits: modelBits + dataBits };
}
function vectorEqualUpToSign(a, b) {
  if (a.length !== b.length) return false;
  const same = a.every((x, i) => x === b[i]);
  const neg = a.every((x, i) => x === -b[i]);
  return same || neg;
}
function asFeatureLawVector(law) { return FEATURES.map(f => law[f]); }
function sceneAsJsonRows(events, artifactsByName, featureList) {
  return events.map(e => ({
    id: e.id,
    scene: e.scene,
    signature: sceneSignature(e.scene),
    outcome: boolLabel(e.coherent),
    hiddenCharge: sceneCharge(e.scene, artifactsByName),
    featureCounts: Object.fromEntries(featureList.map((f, i) => [f, sceneFeatureCounts(e.scene, artifactsByName, featureList)[i]])),
    text: e.text
  }));
}

function makeInterventions(events, allArtifacts, artifactsByName, featureList, q, limit = 30) {
  const fractured = events.filter(e => !e.coherent);
  const candidates = allArtifacts.map(a => ({ name: a.name, q: scoreVector(sceneFeatureCounts({ [a.name]: 1 }, artifactsByName, featureList), q) }));
  const interventions = [];
  for (const e of fractured) {
    if (interventions.length >= limit) break;
    const x = sceneFeatureCounts(e.scene, artifactsByName, featureList);
    const residual = scoreVector(x, q);
    let best = null;
    // single artifact addition
    for (const c of candidates) {
      for (let count = 1; count <= 6; count++) {
        const finalResidual = residual + count * c.q;
        if (finalResidual === 0) {
          best = { additions: { [c.name]: count }, finalResidual };
          break;
        }
      }
      if (best) break;
    }
    // pair addition if single artifact does not work
    if (!best) {
      for (let i = 0; i < candidates.length && !best; i++) {
        for (let j = i; j < candidates.length && !best; j++) {
          for (let c1 = 1; c1 <= 4 && !best; c1++) {
            for (let c2 = 1; c2 <= 4 && !best; c2++) {
              const finalResidual = residual + c1 * candidates[i].q + c2 * candidates[j].q;
              if (finalResidual === 0) {
                best = { additions: { [candidates[i].name]: c1, [candidates[j].name]: (candidates[j].name === candidates[i].name ? c1 + c2 : c2) }, finalResidual };
              }
            }
          }
        }
      }
    }
    if (best) {
      const repaired = Object.assign({}, e.scene);
      for (const [name, count] of Object.entries(best.additions)) repaired[name] = (repaired[name] || 0) + count;
      const oracleFinal = sceneCharge(repaired, artifactsByName);
      interventions.push({
        eventId: e.id,
        originalScene: sceneSignature(e.scene),
        originalOutcome: boolLabel(e.coherent),
        discoveredResidualBefore: residual,
        additions: best.additions,
        repairedScene: sceneSignature(repaired),
        discoveredResidualAfter: best.finalResidual,
        oracleChargeAfter: oracleFinal,
        oracleOutcomeAfter: boolLabel(oracleFinal === 0),
        verified: oracleFinal === 0
      });
    }
  }
  return interventions;
}

function run() {
  ensureDir(DATA_DIR); ensureDir(RESULTS_DIR);
  const rng = mulberry32(SEED);
  const data = generateDataset(rng);
  const allArtifacts = KNOWN_ARTIFACTS.concat(NOVEL_ARTIFACTS);
  const artifactsByName = makeArtifactsByName(allArtifacts);
  const knownByName = makeArtifactsByName(KNOWN_ARTIFACTS);
  const novelByName = makeArtifactsByName(NOVEL_ARTIFACTS);
  const cards = allArtifacts.map(artifactCard);
  const parsedCards = cards.map(parseArtifactCard);
  const parsedArtifactsByName = makeArtifactsByName(parsedCards.map(c => ({ name: c.name, attrs: c.attrs })));

  // Persist the semantic universe and logs used by the experiment.
  writeText(path.join(DATA_DIR, 'artifact_cards.md'), '# Artifact cards\n\n' + cards.map(c => `- ${c}`).join('\n') + '\n');
  writeJSON(path.join(DATA_DIR, 'artifact_cards.json'), allArtifacts.map(a => ({ name: a.name, attributes: a.attrs, hiddenCharge: artifactCharge(a) })));
  writeJSON(path.join(DATA_DIR, 'parsed_artifact_cards.json'), parsedCards);
  writeText(path.join(DATA_DIR, 'train_logs.md'), '# Training logs\n\n' + data.train.map(e => `- ${e.text}`).join('\n') + '\n');
  writeJSON(path.join(DATA_DIR, 'train_events.json'), sceneAsJsonRows(data.train, artifactsByName, FEATURES));
  writeJSON(path.join(DATA_DIR, 'known_test_events.json'), sceneAsJsonRows(data.knownTest, artifactsByName, FEATURES));
  writeJSON(path.join(DATA_DIR, 'extrapolation_events.json'), sceneAsJsonRows(data.extrapolation, artifactsByName, FEATURES));
  writeJSON(path.join(DATA_DIR, 'novel_object_events.json'), sceneAsJsonRows(data.novelObjects, artifactsByName, FEATURES));
  writeJSON(path.join(DATA_DIR, 'hidden_oracle_for_reproducibility.json'), { hiddenLaw: HIDDEN_LAW, note: 'This file is for reproducibility and final verification only. The abductor receives parsed attributes and labels, not this oracle.' });

  // Abductive models.
  const semantic = abductSemanticInvariant(data.train, parsedArtifactsByName, FEATURES);
  const knownObjectNames = KNOWN_ARTIFACTS.map(a => a.name);
  const objectInvariant = abductObjectInvariant(data.train, knownObjectNames);

  // Baselines.
  const baselines = [
    majorityTrain(data.train),
    exactSceneMemorizer(data.train),
    nearestNeighbor(data.train, parsedArtifactsByName, FEATURES, 3),
    logisticFeatureClassifier(data.train, parsedArtifactsByName, FEATURES)
  ];

  const splits = {
    train: data.train,
    known_test: data.knownTest,
    extrapolation: data.extrapolation,
    novel_objects: data.novelObjects
  };

  const metrics = [];
  function addMetric(modelName, split, accuracy, extra = {}) {
    metrics.push({ model: modelName, split, accuracy: Number(accuracy.toFixed(6)), ...extra });
  }

  for (const model of baselines) {
    for (const [split, events] of Object.entries(splits)) {
      const ev = evaluateModel(model, events);
      addMetric(model.name, split, ev.accuracy);
    }
  }

  if (semantic.success) {
    const pred = e => predictSemantic(e, parsedArtifactsByName, FEATURES, semantic.q);
    for (const [split, events] of Object.entries(splits)) addMetric('Semantic abductive invariant', split, accuracyFromPredict(events, pred));
  }

  if (objectInvariant.success) {
    const pred = e => predictObject(e, knownObjectNames, objectInvariant.q);
    for (const [split, events] of Object.entries(splits)) addMetric('Object-id invariant', split, accuracyFromPredict(events, pred));
  }

  // VSA/HDC residual version of the discovered semantic concept.
  const hdcRng = mulberry32(SEED + 1);
  const hv = makeHypervectors(FEATURES, HDC_DIM, hdcRng);
  const artifactVectors = buildArtifactVectors(allArtifacts.map(a => ({ name: a.name, attrs: parsedArtifactsByName[a.name].attrs })), hv, HDC_DIM);
  let hdcSummary = null;
  if (semantic.success) {
    const conceptVec = encodeConcept(semantic.q, FEATURES, hv, HDC_DIM);
    const trainScores = data.train.map(e => hdcResidual(e.scene, artifactVectors, conceptVec, HDC_DIM));
    const trainLabels = data.train.map(e => e.coherent);
    const thresholdFit = fitThreshold(trainScores, trainLabels);
    hdcSummary = { dimension: HDC_DIM, threshold: thresholdFit.threshold, trainThresholdAccuracy: thresholdFit.accuracy };
    for (const [split, events] of Object.entries(splits)) {
      const ev = evaluateHDC(events, artifactVectors, conceptVec, HDC_DIM, thresholdFit.threshold);
      addMetric('HDC residual threshold', split, ev.accuracy, { threshold: Number(thresholdFit.threshold.toFixed(6)) });
    }
  }

  // Negative control: shuffle training labels and run the same abductor on corrupted evidence.
  const shuffledLabels = shuffle(mulberry32(SEED + 7), data.train.map(e => e.coherent));
  const shuffledTrain = data.train.map((e, i) => ({ ...e, coherent: shuffledLabels[i] }));
  const shuffledSemantic = abductSemanticInvariant(shuffledTrain, parsedArtifactsByName, FEATURES);
  let negativeControl = {
    protocol: 'Training labels were permuted while artifact cards and event scenes were kept fixed. The same semantic abductor was then run on the corrupted evidence and evaluated against the true oracle-labeled holdout sets.',
    shuffledSemantic
  };
  if (shuffledSemantic.success) {
    const pred = e => predictSemantic(e, parsedArtifactsByName, FEATURES, shuffledSemantic.q);
    negativeControl.trueTrainAccuracy = accuracyFromPredict(data.train, pred);
    negativeControl.trueKnownTestAccuracy = accuracyFromPredict(data.knownTest, pred);
    negativeControl.trueNovelObjectAccuracy = accuracyFromPredict(data.novelObjects, pred);
    negativeControl.sameAsHiddenUpToSign = vectorEqualUpToSign(shuffledSemantic.q, asFeatureLawVector(HIDDEN_LAW));
  } else {
    negativeControl.trueTrainAccuracy = null;
    negativeControl.trueKnownTestAccuracy = null;
    negativeControl.trueNovelObjectAccuracy = null;
    negativeControl.sameAsHiddenUpToSign = false;
  }

  // Interventions on held-out fractured events, mixing known and novel object names.
  const interventionPool = data.extrapolation.concat(data.novelObjects);
  const interventions = semantic.success ? makeInterventions(interventionPool, allArtifacts, parsedArtifactsByName, FEATURES, semantic.q, 30) : [];

  // MDL-like table: accuracy alone is not enough; compare hypothesis class simplicity + residual error.
  const hypothesisRows = [];
  for (const model of baselines) {
    const pred = e => model.predict(e);
    const mistakes = mistakesFromPredict(data.train, pred);
    const modelDetails = model.model || {};
    const mdl = approxMdlBits(model.name, data.train, mistakes, modelDetails);
    hypothesisRows.push({ model: model.name, trainMistakes: mistakes, ...mdl, explanation: 'Baseline predictor; does not introduce a conservation-like latent property.' });
  }
  if (objectInvariant.success) {
    const pred = e => predictObject(e, knownObjectNames, objectInvariant.q);
    const mistakes = mistakesFromPredict(data.train, pred);
    const mdl = approxMdlBits('Object-id invariant', data.train, mistakes, objectInvariant);
    hypothesisRows.push({ model: 'Object-id invariant', trainMistakes: mistakes, ...mdl, explanation: 'Finds charges for known object names; cannot infer charges for new objects from descriptions.' });
  }
  if (semantic.success) {
    const pred = e => predictSemantic(e, parsedArtifactsByName, FEATURES, semantic.q);
    const mistakes = mistakesFromPredict(data.train, pred);
    const mdl = approxMdlBits('Semantic abductive invariant', data.train, mistakes, semantic);
    hypothesisRows.push({ model: 'Semantic abductive invariant', trainMistakes: mistakes, ...mdl, explanation: 'Introduces a sparse latent semantic property over words and an equality-to-zero law.' });
    if (hdcSummary) {
      const hdcMistakes = data.train.length - Math.round(metrics.find(m => m.model === 'HDC residual threshold' && m.split === 'train').accuracy * data.train.length);
      const mdlHdc = approxMdlBits('HDC residual threshold', data.train, hdcMistakes, semantic);
      hypothesisRows.push({ model: 'HDC residual threshold', trainMistakes: hdcMistakes, ...mdlHdc, explanation: 'Distributed VSA/HDC realization of the same concept, with a learned threshold.' });
    }
  }
  hypothesisRows.sort((a, b) => a.totalBits - b.totalBits);

  const hiddenVector = asFeatureLawVector(HIDDEN_LAW);
  const summary = {
    repositoryPlaceholder: '[PLACEHOLDER: INSERT GITHUB REPOSITORY URL HERE]',
    seed: SEED,
    hdcDimension: HDC_DIM,
    counts: {
      knownArtifacts: KNOWN_ARTIFACTS.length,
      novelArtifacts: NOVEL_ARTIFACTS.length,
      train: data.train.length,
      knownTest: data.knownTest.length,
      extrapolation: data.extrapolation.length,
      novelObjects: data.novelObjects.length
    },
    features: FEATURES,
    hiddenLaw: HIDDEN_LAW,
    hiddenVector,
    semanticAbduction: semantic,
    objectIdAbduction: objectInvariant,
    hdcSummary,
    conceptRecoveredUpToSign: semantic.success && vectorEqualUpToSign(semantic.q, hiddenVector),
    interventions: {
      attempted: interventions.length,
      verified: interventions.filter(x => x.verified).length
    },
    negativeControl,
    metrics,
    hypothesisRows
  };

  writeJSON(path.join(RESULTS_DIR, 'summary.json'), summary);
  writeJSON(path.join(RESULTS_DIR, 'discovered_concept.json'), semantic);
  writeJSON(path.join(RESULTS_DIR, 'object_invariant.json'), objectInvariant);
  writeJSON(path.join(RESULTS_DIR, 'negative_control.json'), negativeControl);
  writeJSON(path.join(RESULTS_DIR, 'interventions.json'), interventions);
  writeJSON(path.join(RESULTS_DIR, 'hypothesis_comparison.json'), hypothesisRows);

  // CSV outputs for quick inspection.
  const metricHeader = 'model,split,accuracy,threshold\n';
  const metricRows = metrics.map(m => `${JSON.stringify(m.model)},${m.split},${m.accuracy},${m.threshold !== undefined ? m.threshold : ''}`).join('\n');
  writeText(path.join(RESULTS_DIR, 'metrics.csv'), metricHeader + metricRows + '\n');
  const interventionHeader = 'eventId,originalScene,discoveredResidualBefore,additions,repairedScene,discoveredResidualAfter,oracleChargeAfter,verified\n';
  const interventionRows = interventions.map(x => [
    x.eventId,
    JSON.stringify(x.originalScene),
    x.discoveredResidualBefore,
    JSON.stringify(JSON.stringify(x.additions)),
    JSON.stringify(x.repairedScene),
    x.discoveredResidualAfter,
    x.oracleChargeAfter,
    x.verified
  ].join(',')).join('\n');
  writeText(path.join(RESULTS_DIR, 'interventions.csv'), interventionHeader + interventionRows + '\n');
  const hypothesisHeader = 'model,trainMistakes,modelBits,dataBits,totalBits,explanation\n';
  const hypothesisCsv = hypothesisRows.map(h => [JSON.stringify(h.model), h.trainMistakes, h.modelBits.toFixed(3), h.dataBits.toFixed(3), h.totalBits.toFixed(3), JSON.stringify(h.explanation)].join(',')).join('\n');
  writeText(path.join(RESULTS_DIR, 'hypothesis_comparison.csv'), hypothesisHeader + hypothesisCsv + '\n');

  // Human-readable certificate, not just an OK line.
  const weightsLines = FEATURES.map((f, i) => `- ${f}: hidden ${hiddenVector[i]}, discovered ${semantic.success ? semantic.q[i] : 'FAILED'}`).join('\n');
  const metricLines = metrics.map(m => `- ${m.model} / ${m.split}: ${m.accuracy.toFixed(3)}`).join('\n');
  const controlText = negativeControl.shuffledSemantic.success
    ? `The shuffled-label abductor produced ${JSON.stringify(negativeControl.shuffledSemantic.weights)}, which is hidden-law-equivalent: ${negativeControl.sameAsHiddenUpToSign}. Its true known-test accuracy was ${negativeControl.trueKnownTestAccuracy}.`
    : `The shuffled-label abductor failed to find a non-trivial invariant: ${negativeControl.shuffledSemantic.reason}`;
  const cert = `# Abduction certificate\n\nThis file is generated by \`node scripts/run_experiment.js\`. It is intended to make the claim inspectable rather than rhetorical.\n\n## Pre-registered criterion\n\nThe run counts as a successful semantic abduction only if all of the following hold:\n\n1. the abductor receives artifact descriptions and event labels, not the hidden oracle;\n2. it introduces a latent semantic property as a sparse vector over words;\n3. the property classifies held-out known-object events;\n4. it extrapolates to larger counts;\n5. it predicts events made only of novel objects described by words;\n6. it proposes counterfactual repairs that the hidden oracle verifies;\n7. the same method fails or generalizes poorly when labels are shuffled.\n\n## Hidden vs discovered concept\n\n${weightsLines}\n\nConcept recovered up to sign: ${summary.conceptRecoveredUpToSign}\n\n## Metrics\n\n${metricLines}\n\n## Negative control\n\n${controlText}\n\n## Interventions\n\nVerified repairs: ${summary.interventions.verified}/${summary.interventions.attempted}. See \`results/interventions.csv\`.\n\n## Caveat\n\nThe hypothesis class is deliberately restricted to additive semantic invariants. The experiment demonstrates abduction inside that specified class; it does not demonstrate unconstrained autonomous science.\n`;
  writeText(path.join(RESULTS_DIR, 'abduction_certificate.md'), cert);

  console.log(JSON.stringify({
    semanticRecovered: summary.conceptRecoveredUpToSign,
    semanticTrainAccuracy: metrics.find(m => m.model === 'Semantic abductive invariant' && m.split === 'train')?.accuracy,
    semanticNovelAccuracy: metrics.find(m => m.model === 'Semantic abductive invariant' && m.split === 'novel_objects')?.accuracy,
    interventionsVerified: `${summary.interventions.verified}/${summary.interventions.attempted}`,
    shuffledControlFoundInvariant: negativeControl.shuffledSemantic.success,
    summaryFile: 'results/summary.json'
  }, null, 2));
}

run();
