'use strict';

function featureVectorForArtifact(artifact, featureList) {
  const set = new Set(artifact.attrs);
  return featureList.map(f => set.has(f) ? 1 : 0);
}

function sceneFeatureCounts(scene, artifactsByName, featureList) {
  const x = new Array(featureList.length).fill(0);
  for (const [name, count] of Object.entries(scene)) {
    const artifact = artifactsByName[name];
    if (!artifact) continue;
    const set = new Set(artifact.attrs);
    for (let i = 0; i < featureList.length; i++) {
      if (set.has(featureList[i])) x[i] += count;
    }
  }
  return x;
}

function objectCountVector(scene, objectNames) {
  return objectNames.map(name => scene[name] || 0);
}

function scoreVector(x, q) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * q[i];
  return s;
}

function normalizeSign(q, featureList, preferredFeature = 'sunlit') {
  const idx = featureList.indexOf(preferredFeature);
  if (idx >= 0 && q[idx] < 0) return q.map(x => -x);
  const first = q.find(x => x !== 0);
  if (first !== undefined && first < 0) return q.map(x => -x);
  return q.slice();
}

function vectorToObject(q, names) {
  const out = {};
  for (let i = 0; i < names.length; i++) out[names[i]] = q[i];
  return out;
}

function accuracy(preds, labels) {
  let correct = 0;
  for (let i = 0; i < preds.length; i++) if (preds[i] === labels[i]) correct++;
  return correct / labels.length;
}

module.exports = {
  featureVectorForArtifact,
  sceneFeatureCounts,
  objectCountVector,
  scoreVector,
  normalizeSign,
  vectorToObject,
  accuracy
};
