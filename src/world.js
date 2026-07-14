'use strict';

const { choice, randint } = require('./rng');

const FEATURES = [
  'sunlit', 'moonlit', 'iron', 'glass', 'electric', 'organic',
  'red', 'blue', 'ancient', 'silent', 'floating', 'engraved'
];

// This is the hidden oracle used only for synthetic data generation and final verification.
// The abduction code receives only parsed artifact attributes and labeled event logs.
const HIDDEN_LAW = {
  sunlit: 2,
  moonlit: -2,
  iron: 1,
  glass: -1,
  electric: 1,
  organic: -1,
  red: 0,
  blue: 0,
  ancient: 0,
  silent: 0,
  floating: 0,
  engraved: 0
};

const KNOWN_ARTIFACTS = [
  { name: 'Helioforge Key', attrs: ['sunlit', 'iron', 'electric', 'red'], noun: 'relic', note: 'carried by gate pilots' },
  { name: 'Nulltide Lantern', attrs: ['moonlit', 'glass', 'organic', 'blue'], noun: 'lamp', note: 'grown in quiet oceans' },
  { name: 'Solar Needle', attrs: ['sunlit', 'electric', 'ancient'], noun: 'needle', note: 'found inside a collapsed star map' },
  { name: 'Moon Glass', attrs: ['moonlit', 'glass', 'engraved'], noun: 'shard', note: 'etched with tidal coordinates' },
  { name: 'Spark Coin', attrs: ['iron', 'electric', 'red'], noun: 'coin', note: 'used as a battery-token' },
  { name: 'Glass Moth', attrs: ['glass', 'organic', 'floating'], noun: 'moth', note: 'alive only in vacuum gardens' },
  { name: 'Iron Root', attrs: ['iron', 'organic', 'ancient'], noun: 'root', note: 'taken from a metal forest' },
  { name: 'Eclipse Shell', attrs: ['sunlit', 'moonlit', 'silent'], noun: 'shell', note: 'silent during every ignition' },
  { name: 'Sunwire Prism', attrs: ['sunlit', 'glass', 'electric', 'blue'], noun: 'prism', note: 'wired to the navigation altar' },
  { name: 'Nightseed Engine', attrs: ['moonlit', 'organic', 'iron', 'red'], noun: 'engine', note: 'sprouted from a dormant machine' },
  { name: 'Auric Feather', attrs: ['sunlit', 'floating', 'engraved'], noun: 'feather', note: 'engraved with pilot songs' },
  { name: 'Void Bloom', attrs: ['moonlit', 'organic', 'silent'], noun: 'flower', note: 'opens only after failed jumps' },
  { name: 'Wire Orchid', attrs: ['electric', 'organic', 'floating'], noun: 'orchid', note: 'hums near broken clocks' },
  { name: 'Silver Ash', attrs: ['iron', 'glass', 'ancient'], noun: 'ash', note: 'sifted from a forgotten forge' },
  { name: 'Amber Dial', attrs: ['sunlit', 'glass', 'red'], noun: 'dial', note: 'warms when the gate wakes' },
  { name: 'Lunar Fuse', attrs: ['moonlit', 'electric', 'blue'], noun: 'fuse', note: 'cold even when charged' }
];

const NOVEL_ARTIFACTS = [
  { name: 'Daysteel Compass', attrs: ['sunlit', 'iron', 'floating'], noun: 'compass', note: 'not present in the training logs' },
  { name: 'Dreamglass Seed', attrs: ['moonlit', 'glass', 'organic'], noun: 'seed', note: 'not present in the training logs' },
  { name: 'Thunder Reed', attrs: ['electric', 'organic', 'blue'], noun: 'reed', note: 'not present in the training logs' },
  { name: 'Solar Thread', attrs: ['sunlit', 'electric', 'engraved'], noun: 'thread', note: 'not present in the training logs' },
  { name: 'Mist Mirror', attrs: ['moonlit', 'glass', 'silent'], noun: 'mirror', note: 'not present in the training logs' },
  { name: 'Iron Ember', attrs: ['iron', 'red', 'ancient'], noun: 'ember', note: 'not present in the training logs' },
  { name: 'Glass Root', attrs: ['glass', 'organic', 'floating'], noun: 'root', note: 'not present in the training logs' },
  { name: 'Bright Coil', attrs: ['sunlit', 'iron', 'electric'], noun: 'coil', note: 'not present in the training logs' },
  { name: 'Moon Orchid', attrs: ['moonlit', 'organic', 'blue'], noun: 'orchid', note: 'not present in the training logs' },
  { name: 'Static Shell', attrs: ['electric', 'glass', 'engraved'], noun: 'shell', note: 'not present in the training logs' }
];

function artifactCharge(artifact, law = HIDDEN_LAW) {
  return artifact.attrs.reduce((s, a) => s + (law[a] || 0), 0);
}

function sceneCharge(scene, artifactsByName, law = HIDDEN_LAW) {
  let total = 0;
  for (const [name, count] of Object.entries(scene)) {
    total += count * artifactCharge(artifactsByName[name], law);
  }
  return total;
}

function sceneSignature(scene) {
  return Object.entries(scene)
    .filter(([, c]) => c !== 0)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, count]) => `${count}x ${name}`)
    .join('; ');
}

function artifactCard(artifact) {
  const attrs = artifact.attrs.join(', ');
  return `${artifact.name}: a ${attrs} ${artifact.noun}, ${artifact.note}.`;
}

function eventLog(event) {
  return `Eos Gate Log ${String(event.id).padStart(4, '0')}: The crew placed ${sceneSignature(event.scene)} in the gate chamber. Outcome: ${event.coherent ? 'COHERENT' : 'FRACTURED'}.`;
}

function makeArtifactsByName(artifacts) {
  const out = {};
  for (const a of artifacts) out[a.name] = a;
  return out;
}

function makeRandomScene(rng, artifacts, opts) {
  const minTerms = opts.minTerms || 2;
  const maxTerms = opts.maxTerms || 5;
  const minCount = opts.minCount || 1;
  const maxCount = opts.maxCount || 3;
  const terms = randint(rng, minTerms, maxTerms);
  const scene = {};
  for (let i = 0; i < terms; i++) {
    const a = choice(rng, artifacts);
    scene[a.name] = (scene[a.name] || 0) + randint(rng, minCount, maxCount);
  }
  return scene;
}

function generateBalancedEvents(rng, artifacts, targetStable, targetFractured, opts) {
  const artifactsByName = makeArtifactsByName(artifacts);
  const stable = [];
  const fractured = [];
  const seen = new Set();
  let attempts = 0;
  const maxAttempts = opts.maxAttempts || 1000000;
  while ((stable.length < targetStable || fractured.length < targetFractured) && attempts < maxAttempts) {
    attempts++;
    const scene = makeRandomScene(rng, artifacts, opts);
    const sig = sceneSignature(scene);
    if (seen.has(sig)) continue;
    const charge = sceneCharge(scene, artifactsByName);
    const coherent = charge === 0;
    if (coherent && stable.length < targetStable) {
      stable.push({ id: 0, scene, coherent, hiddenCharge: charge, text: '' });
      seen.add(sig);
    } else if (!coherent && fractured.length < targetFractured) {
      fractured.push({ id: 0, scene, coherent, hiddenCharge: charge, text: '' });
      seen.add(sig);
    }
  }
  if (stable.length < targetStable || fractured.length < targetFractured) {
    throw new Error(`Could not generate enough events after ${attempts} attempts: stable ${stable.length}/${targetStable}, fractured ${fractured.length}/${targetFractured}`);
  }
  const merged = stable.concat(fractured);
  for (let i = 0; i < merged.length; i++) {
    merged[i].id = i + 1;
    merged[i].text = eventLog(merged[i]);
  }
  return merged;
}

function generateDataset(rng) {
  const train = generateBalancedEvents(rng, KNOWN_ARTIFACTS, 80, 80, { minTerms: 2, maxTerms: 5, minCount: 1, maxCount: 3 });
  const knownTest = generateBalancedEvents(rng, KNOWN_ARTIFACTS, 150, 150, { minTerms: 2, maxTerms: 5, minCount: 1, maxCount: 3 });
  const extrapolation = generateBalancedEvents(rng, KNOWN_ARTIFACTS, 150, 150, { minTerms: 2, maxTerms: 5, minCount: 3, maxCount: 8 });
  const novelObjects = generateBalancedEvents(rng, NOVEL_ARTIFACTS, 150, 150, { minTerms: 2, maxTerms: 5, minCount: 1, maxCount: 4 });
  return { train, knownTest, extrapolation, novelObjects };
}

module.exports = {
  FEATURES,
  HIDDEN_LAW,
  KNOWN_ARTIFACTS,
  NOVEL_ARTIFACTS,
  artifactCharge,
  sceneCharge,
  sceneSignature,
  artifactCard,
  eventLog,
  makeArtifactsByName,
  generateDataset,
  generateBalancedEvents
};
