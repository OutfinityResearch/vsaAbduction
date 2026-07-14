#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const summaryPath = path.join(ROOT, 'results', 'summary.json');
if (!fs.existsSync(summaryPath)) {
  console.error('Missing results/summary.json. Run `npm run run` first.');
  process.exit(1);
}
const s = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
const metric = (model, split) => {
  const rec = s.metrics.find(m => m.model === model && m.split === split);
  if (!rec) throw new Error(`Missing metric: ${model} / ${split}`);
  return rec.accuracy;
};
const failures = [];
function check(cond, msg) { if (!cond) failures.push(msg); }
check(s.semanticAbduction.success, 'semantic abductor should return a concept');
check(s.conceptRecoveredUpToSign === true, 'semantic concept should match hidden oracle up to sign/scale');
check(metric('Semantic abductive invariant', 'train') === 1, 'semantic invariant should fit training data');
check(metric('Semantic abductive invariant', 'known_test') === 1, 'semantic invariant should predict held-out known-object events');
check(metric('Semantic abductive invariant', 'extrapolation') === 1, 'semantic invariant should extrapolate to larger counts');
check(metric('Semantic abductive invariant', 'novel_objects') === 1, 'semantic invariant should predict novel objects from descriptions');
check(metric('Object-id invariant', 'novel_objects') <= 0.55, 'object-id invariant should fail on novel objects');
check(s.interventions.verified === s.interventions.attempted && s.interventions.attempted >= 20, 'counterfactual repairs should be oracle-verified');
check(s.negativeControl.sameAsHiddenUpToSign === false, 'shuffled-label control should not recover hidden concept');
check(s.negativeControl.shuffledSemantic.success === false || (s.negativeControl.trueKnownTestAccuracy !== null && s.negativeControl.trueKnownTestAccuracy < 0.65), 'shuffled-label control should fail or generalize poorly');
if (failures.length) {
  console.error('Verification failed:');
  for (const f of failures) console.error(' - ' + f);
  process.exit(1);
}
console.log('Verification report');
console.log('-------------------');
console.log(`Semantic invariant recovered: ${s.conceptRecoveredUpToSign}`);
console.log(`Train / known / extrapolation / novel accuracies: ${metric('Semantic abductive invariant','train')}, ${metric('Semantic abductive invariant','known_test')}, ${metric('Semantic abductive invariant','extrapolation')}, ${metric('Semantic abductive invariant','novel_objects')}`);
console.log(`Object-id invariant on novel objects: ${metric('Object-id invariant','novel_objects')}`);
console.log(`HDC residual on extrapolation: ${metric('HDC residual threshold','extrapolation')}`);
console.log(`Interventions verified: ${s.interventions.verified}/${s.interventions.attempted}`);
console.log(`Shuffled-label control recovered hidden concept: ${s.negativeControl.sameAsHiddenUpToSign}`);
