'use strict';

const { sceneFeatureCounts, scoreVector } = require('./features');

function makeHypervectors(featureList, dim, rng) {
  const hv = {};
  for (const f of featureList) {
    const arr = new Int8Array(dim);
    for (let i = 0; i < dim; i++) arr[i] = rng() < 0.5 ? -1 : 1;
    hv[f] = arr;
  }
  return hv;
}

function encodeArtifact(attrs, hv, dim) {
  const out = new Int16Array(dim);
  for (const a of attrs) {
    const v = hv[a];
    for (let i = 0; i < dim; i++) out[i] += v[i];
  }
  return out;
}

function encodeConcept(q, featureList, hv, dim) {
  const out = new Int16Array(dim);
  for (let j = 0; j < featureList.length; j++) {
    const w = q[j];
    if (w === 0) continue;
    const v = hv[featureList[j]];
    for (let i = 0; i < dim; i++) out[i] += w * v[i];
  }
  return out;
}

function encodeScene(scene, artifactVectors, dim) {
  const out = new Int32Array(dim);
  for (const [name, count] of Object.entries(scene)) {
    const v = artifactVectors[name];
    if (!v) continue;
    for (let i = 0; i < dim; i++) out[i] += count * v[i];
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function hdcResidual(scene, artifactVectors, conceptVec, dim) {
  const sceneVec = encodeScene(scene, artifactVectors, dim);
  return dot(sceneVec, conceptVec) / dim;
}

function fitThreshold(scores, labels) {
  const absScores = scores.map(Math.abs).sort((a, b) => a - b);
  const candidates = [0];
  for (let i = 0; i < absScores.length - 1; i++) {
    candidates.push((absScores[i] + absScores[i + 1]) / 2);
  }
  candidates.push(absScores[absScores.length - 1] + 1e-9);
  let best = { threshold: 0, accuracy: -1 };
  for (const t of candidates) {
    let correct = 0;
    for (let i = 0; i < scores.length; i++) {
      const pred = Math.abs(scores[i]) <= t;
      if (pred === labels[i]) correct++;
    }
    const acc = correct / scores.length;
    if (acc > best.accuracy) best = { threshold: t, accuracy: acc };
  }
  return best;
}

function evaluateHDC(events, artifactVectors, conceptVec, dim, threshold) {
  let correct = 0;
  const rows = [];
  for (const e of events) {
    const residual = hdcResidual(e.scene, artifactVectors, conceptVec, dim);
    const pred = Math.abs(residual) <= threshold;
    if (pred === e.coherent) correct++;
    rows.push({ id: e.id, residual, pred, coherent: e.coherent });
  }
  return { accuracy: correct / events.length, rows };
}

function buildArtifactVectors(artifacts, hv, dim) {
  const out = {};
  for (const a of artifacts) out[a.name] = encodeArtifact(a.attrs, hv, dim);
  return out;
}

module.exports = {
  makeHypervectors,
  encodeArtifact,
  encodeConcept,
  encodeScene,
  hdcResidual,
  fitThreshold,
  evaluateHDC,
  buildArtifactVectors
};
