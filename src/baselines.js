'use strict';

const { sceneSignature } = require('./world');
const { sceneFeatureCounts, scoreVector } = require('./features');

function majorityTrain(trainEvents) {
  const positives = trainEvents.filter(e => e.coherent).length;
  const prediction = positives >= trainEvents.length - positives;
  return {
    name: 'Majority label',
    predict: () => prediction,
    model: { prediction }
  };
}

function exactSceneMemorizer(trainEvents) {
  const counts = new Map();
  for (const e of trainEvents) {
    const sig = sceneSignature(e.scene);
    if (!counts.has(sig)) counts.set(sig, { true: 0, false: 0 });
    counts.get(sig)[String(e.coherent)]++;
  }
  const majority = majorityTrain(trainEvents).model.prediction;
  return {
    name: 'Exact scene memorizer',
    predict: e => {
      const rec = counts.get(sceneSignature(e.scene));
      if (!rec) return majority;
      return rec.true >= rec.false;
    },
    model: { uniqueScenes: counts.size, defaultPrediction: majority }
  };
}

function distanceSquared(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function nearestNeighbor(trainEvents, artifactsByName, featureList, k = 3) {
  const train = trainEvents.map(e => ({ x: sceneFeatureCounts(e.scene, artifactsByName, featureList), y: e.coherent }));
  return {
    name: `${k}-nearest neighbors on feature counts`,
    predict: e => {
      const x = sceneFeatureCounts(e.scene, artifactsByName, featureList);
      const ranked = train.map(t => ({ y: t.y, d: distanceSquared(x, t.x) })).sort((a, b) => a.d - b.d).slice(0, k);
      const positives = ranked.filter(r => r.y).length;
      return positives >= Math.ceil(k / 2);
    },
    model: { k, trainingVectors: train.length }
  };
}

function sigmoid(z) {
  if (z < -35) return 0;
  if (z > 35) return 1;
  return 1 / (1 + Math.exp(-z));
}

function logisticFeatureClassifier(trainEvents, artifactsByName, featureList, opts = {}) {
  const lr = opts.lr || 0.003;
  const epochs = opts.epochs || 800;
  const l2 = opts.l2 || 0.0005;
  const train = trainEvents.map(e => ({ x: sceneFeatureCounts(e.scene, artifactsByName, featureList), y: e.coherent ? 1 : 0 }));
  const w = new Array(featureList.length).fill(0);
  let b = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const item of train) {
      const z = scoreVector(item.x, w) + b;
      const p = sigmoid(z);
      const err = p - item.y;
      for (let j = 0; j < w.length; j++) {
        w[j] -= lr * (err * item.x[j] + l2 * w[j]);
      }
      b -= lr * err;
    }
  }
  return {
    name: 'Logistic feature-bag classifier',
    predict: e => sigmoid(scoreVector(sceneFeatureCounts(e.scene, artifactsByName, featureList), w) + b) >= 0.5,
    model: { weights: Object.fromEntries(featureList.map((f, i) => [f, w[i]])), bias: b, lr, epochs, l2 }
  };
}

function evaluateModel(model, events) {
  let correct = 0;
  const predictions = [];
  for (const e of events) {
    const pred = model.predict(e);
    predictions.push(pred);
    if (pred === e.coherent) correct++;
  }
  return { accuracy: correct / events.length, correct, total: events.length, predictions };
}

module.exports = {
  majorityTrain,
  exactSceneMemorizer,
  nearestNeighbor,
  logisticFeatureClassifier,
  evaluateModel
};
