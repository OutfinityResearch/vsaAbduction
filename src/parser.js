'use strict';

// Deterministic semantic parser for the synthetic micro-world.
// In a full LLM+VSA system, this module would be replaced by an LLM parser.
// Here we keep it deterministic to make the experiment reproducible and auditable.

const { FEATURES } = require('./world');

function parseArtifactCard(text) {
  const [namePart] = text.split(':');
  const lower = text.toLowerCase();
  const attrs = FEATURES.filter(f => lower.includes(f));
  return { name: namePart.trim(), attrs };
}

function parseEventLog(text, artifactNames) {
  const coherent = /Outcome:\s*COHERENT/i.test(text);
  const scene = {};
  for (const name of artifactNames) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`(\\d+)x\\s+${escaped}`, 'i');
    const match = text.match(re);
    if (match) scene[name] = Number(match[1]);
  }
  return { scene, coherent };
}

module.exports = { parseArtifactCard, parseEventLog };
