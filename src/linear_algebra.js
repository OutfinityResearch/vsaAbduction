'use strict';

function cloneMatrix(A) {
  return A.map(row => row.slice());
}

function rref(A, eps = 1e-10) {
  if (A.length === 0) return { matrix: [], pivots: [], rank: 0 };
  const M = cloneMatrix(A);
  const rows = M.length;
  const cols = M[0].length;
  const pivots = [];
  let r = 0;
  for (let c = 0; c < cols && r < rows; c++) {
    let pivot = r;
    let maxAbs = Math.abs(M[r][c]);
    for (let i = r + 1; i < rows; i++) {
      const val = Math.abs(M[i][c]);
      if (val > maxAbs) {
        maxAbs = val;
        pivot = i;
      }
    }
    if (maxAbs < eps) continue;
    if (pivot !== r) {
      const tmp = M[pivot];
      M[pivot] = M[r];
      M[r] = tmp;
    }
    const divisor = M[r][c];
    for (let j = c; j < cols; j++) M[r][j] /= divisor;
    for (let i = 0; i < rows; i++) {
      if (i === r) continue;
      const factor = M[i][c];
      if (Math.abs(factor) < eps) continue;
      for (let j = c; j < cols; j++) M[i][j] -= factor * M[r][j];
    }
    pivots.push(c);
    r++;
  }
  return { matrix: M, pivots, rank: pivots.length };
}

function nullspaceBasis(A, eps = 1e-10) {
  if (A.length === 0) throw new Error('Cannot compute nullspace of empty matrix.');
  const cols = A[0].length;
  const rr = rref(A, eps);
  const pivotSet = new Set(rr.pivots);
  const freeCols = [];
  for (let c = 0; c < cols; c++) {
    if (!pivotSet.has(c)) freeCols.push(c);
  }
  const basis = [];
  for (const f of freeCols) {
    const v = new Array(cols).fill(0);
    v[f] = 1;
    for (let i = 0; i < rr.pivots.length; i++) {
      const p = rr.pivots[i];
      v[p] = -rr.matrix[i][f];
    }
    basis.push(v);
  }
  return { basis, rank: rr.rank, nullity: freeCols.length, pivots: rr.pivots, freeCols };
}

function gcd2(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function gcdArray(arr) {
  let g = 0;
  for (const x of arr) {
    if (x !== 0) g = g === 0 ? Math.abs(x) : gcd2(g, Math.abs(x));
  }
  return g || 1;
}

function vectorDot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function vecNorm(v) {
  return Math.sqrt(vectorDot(v, v));
}

function normalizeSmallIntegerVector(v, opts = {}) {
  const eps = opts.eps || 1e-8;
  const maxCoeff = opts.maxCoeff || 6;
  const nonzero = v.map(Math.abs).filter(x => x > eps);
  if (nonzero.length === 0) return null;
  let best = null;
  // Try multiple scales and keep the rounded vector closest to the continuous direction.
  for (let scale = 0.25; scale <= 12.0001; scale += 0.005) {
    let q = v.map(x => Math.abs(x) < eps ? 0 : Math.round(x * scale));
    if (q.every(x => x === 0)) continue;
    if (q.some(x => Math.abs(x) > maxCoeff)) continue;
    const g = gcdArray(q);
    q = q.map(x => x / g);
    if (q.every(x => x === 0)) continue;
    const qNorm = vecNorm(q);
    const vNorm = vecNorm(v);
    if (qNorm < eps || vNorm < eps) continue;
    const cos = Math.abs(vectorDot(q, v) / (qNorm * vNorm));
    const complexity = q.filter(x => x !== 0).length + 0.05 * q.reduce((s, x) => s + Math.abs(x), 0);
    const score = cos - 0.0001 * complexity;
    if (!best || score > best.score) best = { q, cos, complexity, score };
  }
  return best ? best.q : null;
}

module.exports = { rref, nullspaceBasis, normalizeSmallIntegerVector, gcdArray, vectorDot, vecNorm };
