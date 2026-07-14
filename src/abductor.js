'use strict';

const { nullspaceBasis, normalizeSmallIntegerVector, vectorDot, vecNorm } = require('./linear_algebra');
const { sceneFeatureCounts, objectCountVector, scoreVector, normalizeSign, vectorToObject } = require('./features');

function candidateFromBasis(basisInfo, stableRows, allRows, labels, names, opts = {}) {
  if (basisInfo.nullity === 0) {
    return {
      success: false,
      reason: 'No non-trivial nullspace: the rows labeled COHERENT do not admit a linear invariant in this representation.',
      rank: basisInfo.rank,
      nullity: basisInfo.nullity
    };
  }

  let candidates = [];
  if (basisInfo.nullity === 1) {
    candidates.push(basisInfo.basis[0]);
  } else {
    // If the coherent observations leave a multi-dimensional nullspace, search small
    // combinations of basis vectors and choose the candidate that separates FRACTURED logs.
    const coeffs = [-3, -2, -1, 0, 1, 2, 3];
    const rec = (idx, current, nonzeroCount) => {
      if (idx === basisInfo.basis.length) {
        if (nonzeroCount > 0) candidates.push(current.slice());
        return;
      }
      for (const c of coeffs) {
        const next = current.slice();
        if (c !== 0) {
          const b = basisInfo.basis[idx];
          for (let j = 0; j < next.length; j++) next[j] += c * b[j];
        }
        rec(idx + 1, next, nonzeroCount + (c !== 0 ? 1 : 0));
      }
    };
    rec(0, new Array(names.length).fill(0), 0);
    if (candidates.length > 50000) candidates = candidates.slice(0, 50000);
  }

  let best = null;
  for (const continuous of candidates) {
    let q = normalizeSmallIntegerVector(continuous, { maxCoeff: opts.maxCoeff || 8 });
    if (!q) continue;
    q = normalizeSign(q, names, opts.preferredPositive || names[0]);
    const evalResult = evaluateInvariantRows(q, allRows, labels);
    const nonzero = q.filter(x => x !== 0).length;
    const l1 = q.reduce((s, x) => s + Math.abs(x), 0);
    const stableResidual = meanAbs(stableRows.map(x => scoreVector(x, q)));
    const directionNorm = vecNorm(continuous);
    const roundCos = directionNorm > 0 && vecNorm(q) > 0 ? Math.abs(vectorDot(continuous, q) / (directionNorm * vecNorm(q))) : 0;
    const score = evalResult.accuracy - 0.0005 * nonzero - 0.00005 * l1 - 0.01 * stableResidual + 0.0001 * roundCos;
    if (!best || score > best.selectionScore) {
      best = { q, evalResult, nonzero, l1, stableResidual, roundCos, selectionScore: score };
    }
  }
  if (!best) {
    return {
      success: false,
      reason: 'Nullspace exists but no rounded small-integer invariant could be decoded.',
      rank: basisInfo.rank,
      nullity: basisInfo.nullity
    };
  }
  return {
    success: true,
    q: best.q,
    weights: vectorToObject(best.q, names),
    rank: basisInfo.rank,
    nullity: basisInfo.nullity,
    trainAccuracy: best.evalResult.accuracy,
    trainMistakes: best.evalResult.mistakes,
    stableResidual: best.stableResidual,
    nonzero: best.nonzero,
    l1: best.l1,
    roundCos: best.roundCos,
    selectionScore: best.selectionScore
  };
}

function meanAbs(xs) {
  if (xs.length === 0) return 0;
  return xs.reduce((s, x) => s + Math.abs(x), 0) / xs.length;
}

function evaluateInvariantRows(q, rows, labels) {
  let correct = 0;
  let mistakes = 0;
  const residuals = [];
  for (let i = 0; i < rows.length; i++) {
    const residual = scoreVector(rows[i], q);
    const pred = residual === 0;
    residuals.push(residual);
    if (pred === labels[i]) correct++; else mistakes++;
  }
  return { accuracy: correct / rows.length, mistakes, residuals };
}

function abductSemanticInvariant(events, artifactsByName, featureList, opts = {}) {
  const rows = events.map(e => sceneFeatureCounts(e.scene, artifactsByName, featureList));
  const labels = events.map(e => e.coherent);
  const stableRows = rows.filter((_, i) => labels[i]);
  const basisInfo = nullspaceBasis(stableRows);
  const out = candidateFromBasis(basisInfo, stableRows, rows, labels, featureList, { preferredPositive: 'sunlit', maxCoeff: opts.maxCoeff || 8 });
  out.representation = 'semantic_features';
  out.featureList = featureList.slice();
  return out;
}

function abductObjectInvariant(events, objectNames, opts = {}) {
  const rows = events.map(e => objectCountVector(e.scene, objectNames));
  const labels = events.map(e => e.coherent);
  const stableRows = rows.filter((_, i) => labels[i]);
  const basisInfo = nullspaceBasis(stableRows);
  const out = candidateFromBasis(basisInfo, stableRows, rows, labels, objectNames, { preferredPositive: objectNames[0], maxCoeff: opts.maxCoeff || 16 });
  out.representation = 'object_ids';
  out.objectNames = objectNames.slice();
  return out;
}

function predictSemantic(event, artifactsByName, featureList, q) {
  const x = sceneFeatureCounts(event.scene, artifactsByName, featureList);
  return scoreVector(x, q) === 0;
}

function predictObject(event, objectNames, q) {
  const x = objectCountVector(event.scene, objectNames);
  return scoreVector(x, q) === 0;
}

module.exports = {
  abductSemanticInvariant,
  abductObjectInvariant,
  predictSemantic,
  predictObject,
  evaluateInvariantRows
};
