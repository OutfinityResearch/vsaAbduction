#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const FIG = path.join(ROOT, 'figures');
const RESULTS = path.join(ROOT, 'results');
fs.mkdirSync(FIG, { recursive: true });

function readJSON(file) { return JSON.parse(fs.readFileSync(file, 'utf8')); }
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function write(file, s) { fs.writeFileSync(path.join(FIG, file), s); }

const summary = readJSON(path.join(RESULTS, 'summary.json'));
const metrics = summary.metrics;
const features = summary.features;
const discovered = summary.semanticAbduction.weights;
const hidden = summary.hiddenLaw;
const hypotheses = summary.hypothesisRows;
const interventions = readJSON(path.join(RESULTS, 'interventions.json'));

function conceptWeightsSvg() {
  const width = 980, height = 520, margin = { left: 150, right: 40, top: 50, bottom: 60 };
  const maxAbs = 2.5;
  const chartW = width - margin.left - margin.right;
  const rowH = (height - margin.top - margin.bottom) / features.length;
  const zeroX = margin.left + chartW / 2;
  let body = `<rect width="100%" height="100%" fill="#ffffff"/>`;
  body += `<text x="${width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Discovered latent semantic property</text>`;
  body += `<line x1="${zeroX}" y1="${margin.top-10}" x2="${zeroX}" y2="${height-margin.bottom+8}" stroke="#333" stroke-width="1"/>`;
  for (let i = 0; i < features.length; i++) {
    const f = features[i];
    const w = discovered[f];
    const y = margin.top + i * rowH + rowH * 0.5;
    const barW = Math.abs(w) / maxAbs * (chartW / 2);
    const x = w >= 0 ? zeroX : zeroX - barW;
    body += `<text x="${margin.left-12}" y="${y+5}" text-anchor="end" font-family="Arial" font-size="14">${esc(f)}</text>`;
    body += `<rect x="${x}" y="${y-rowH*0.28}" width="${barW}" height="${rowH*0.56}" fill="${w >= 0 ? '#4c78a8' : '#f58518'}"/>`;
    body += `<text x="${w >= 0 ? x+barW+8 : x-8}" y="${y+5}" text-anchor="${w >= 0 ? 'start' : 'end'}" font-family="Arial" font-size="13">${w}</text>`;
    if (hidden[f] !== w) body += `<text x="${width-10}" y="${y+5}" text-anchor="end" font-family="Arial" font-size="12" fill="red">hidden ${hidden[f]}</text>`;
  }
  body += `<text x="${zeroX}" y="${height-24}" text-anchor="middle" font-family="Arial" font-size="13">Positive and negative semantic contributions; irrelevant descriptors receive zero weight.</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${body}</svg>`;
}

function accuracySvg() {
  const models = [...new Set(metrics.map(m => m.model))];
  const splits = ['train', 'known_test', 'extrapolation', 'novel_objects'];
  const width = 1180, height = 580, margin = { left: 190, right: 30, top: 60, bottom: 110 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;
  const groupW = chartW / splits.length;
  const barW = groupW / (models.length + 1);
  const colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2', '#ff9da6'];
  let body = `<rect width="100%" height="100%" fill="#ffffff"/>`;
  body += `<text x="${width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Generalization by method and split</text>`;
  for (let t = 0; t <= 10; t++) {
    const y = margin.top + chartH - (t/10)*chartH;
    body += `<line x1="${margin.left}" y1="${y}" x2="${width-margin.right}" y2="${y}" stroke="#e6e6e6"/>`;
    body += `<text x="${margin.left-8}" y="${y+4}" text-anchor="end" font-family="Arial" font-size="12">${(t/10).toFixed(1)}</text>`;
  }
  for (let s = 0; s < splits.length; s++) {
    const gx = margin.left + s * groupW;
    body += `<text x="${gx+groupW/2}" y="${height-70}" text-anchor="middle" font-family="Arial" font-size="13">${esc(splits[s].replace('_',' '))}</text>`;
    for (let m = 0; m < models.length; m++) {
      const rec = metrics.find(x => x.model === models[m] && x.split === splits[s]);
      const acc = rec ? rec.accuracy : 0;
      const h = acc * chartH;
      const x = gx + (m+0.5)*barW;
      const y = margin.top + chartH - h;
      body += `<rect x="${x}" y="${y}" width="${barW*0.75}" height="${h}" fill="${colors[m % colors.length]}"/>`;
    }
  }
  let lx = margin.left, ly = height - 45;
  for (let m = 0; m < models.length; m++) {
    body += `<rect x="${lx}" y="${ly-10}" width="12" height="12" fill="${colors[m%colors.length]}"/>`;
    body += `<text x="${lx+18}" y="${ly}" font-family="Arial" font-size="11">${esc(models[m])}</text>`;
    lx += 160;
    if (lx > width - 210) { lx = margin.left; ly += 18; }
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${body}</svg>`;
}

function pipelineSvg() {
  const width = 1180, height = 360;
  const boxes = [
    ['Artifact cards\nand event logs', 'Textual SF micro-world'],
    ['Deterministic parser', 'Extract words and event labels'],
    ['VSA/HDC encoding', 'Bundle words into vectors'],
    ['Abductive invariant', 'Find sparse latent property'],
    ['Evaluation', 'Prediction, novelty, repair, control']
  ];
  let body = `<rect width="100%" height="100%" fill="#ffffff"/>`;
  body += `<text x="${width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Experiment pipeline</text>`;
  const boxW = 190, boxH = 110, gap = 35, y = 105;
  let x = 45;
  for (let i = 0; i < boxes.length; i++) {
    body += `<rect x="${x}" y="${y}" rx="14" ry="14" width="${boxW}" height="${boxH}" fill="#f7f7f7" stroke="#333" stroke-width="1.5"/>`;
    const titleLines = boxes[i][0].split('\n');
    for (let li = 0; li < titleLines.length; li++) {
      body += `<text x="${x+boxW/2}" y="${y+34+li*18}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">${esc(titleLines[li])}</text>`;
    }
    body += `<text x="${x+boxW/2}" y="${y+84}" text-anchor="middle" font-family="Arial" font-size="12" fill="#444">${esc(boxes[i][1])}</text>`;
    if (i < boxes.length - 1) {
      const ax1 = x + boxW + 4, ax2 = x + boxW + gap - 8, ay = y + boxH/2;
      body += `<line x1="${ax1}" y1="${ay}" x2="${ax2}" y2="${ay}" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>`;
    }
    x += boxW + gap;
  }
  body += `<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/></marker></defs>`;
  body += `<text x="${width/2}" y="285" text-anchor="middle" font-family="Arial" font-size="13">The hidden oracle is used only before and after the abductor: before, to generate labels; after, to verify held-out predictions and repairs.</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${body}</svg>`;
}

function hypothesisSvg() {
  const width = 980, height = 440, margin = { left: 210, right: 30, top: 50, bottom: 50 };
  const rows = hypotheses.slice().sort((a, b) => a.totalBits - b.totalBits);
  const maxBits = Math.max(...rows.map(r => r.totalBits));
  const chartW = width - margin.left - margin.right;
  const rowH = (height - margin.top - margin.bottom) / rows.length;
  let body = `<rect width="100%" height="100%" fill="#ffffff"/>`;
  body += `<text x="${width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Approximate MDL-style hypothesis comparison</text>`;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const y = margin.top + i * rowH + rowH * 0.5;
    const w = r.totalBits / maxBits * chartW;
    body += `<text x="${margin.left-10}" y="${y+5}" text-anchor="end" font-family="Arial" font-size="13">${esc(r.model)}</text>`;
    body += `<rect x="${margin.left}" y="${y-rowH*0.25}" width="${w}" height="${rowH*0.5}" fill="#4c78a8"/>`;
    body += `<text x="${margin.left+w+7}" y="${y+5}" font-family="Arial" font-size="12">${r.totalBits.toFixed(1)} bits</text>`;
  }
  body += `<text x="${width/2}" y="${height-20}" text-anchor="middle" font-family="Arial" font-size="12">Lower is better in this coarse diagnostic: compact model plus few residual mistakes.</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${body}</svg>`;
}

function interventionsSvg() {
  const width = 1100, height = 440, margin = { left: 45, right: 45, top: 55, bottom: 40 };
  const rows = interventions.slice(0, 8);
  const rowH = 38;
  let body = `<rect width="100%" height="100%" fill="#ffffff"/>`;
  body += `<text x="${width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Counterfactual repairs proposed by the discovered concept</text>`;
  const headers = ['Before residual', 'Addition', 'After residual', 'Oracle verified'];
  const xs = [160, 470, 750, 945];
  for (let h = 0; h < headers.length; h++) body += `<text x="${xs[h]}" y="${margin.top}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">${headers[h]}</text>`;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const y = margin.top + 25 + i * rowH;
    const add = Object.entries(r.additions).map(([k,v]) => `${v}x ${k}`).join('; ');
    body += `<line x1="${margin.left}" y1="${y-18}" x2="${width-margin.right}" y2="${y-18}" stroke="#eee"/>`;
    body += `<text x="${xs[0]}" y="${y+5}" text-anchor="middle" font-family="Arial" font-size="13">${r.discoveredResidualBefore}</text>`;
    body += `<text x="${xs[1]}" y="${y+5}" text-anchor="middle" font-family="Arial" font-size="12">${esc(add)}</text>`;
    body += `<text x="${xs[2]}" y="${y+5}" text-anchor="middle" font-family="Arial" font-size="13">${r.discoveredResidualAfter}</text>`;
    body += `<text x="${xs[3]}" y="${y+5}" text-anchor="middle" font-family="Arial" font-size="13">${r.verified ? 'yes' : 'no'}</text>`;
  }
  body += `<text x="${width/2}" y="${height-18}" text-anchor="middle" font-family="Arial" font-size="12">A pure classifier predicts labels; a concept can propose a repair that changes the world state.</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${body}</svg>`;
}

write('concept_weights.svg', conceptWeightsSvg());
write('accuracy_bars.svg', accuracySvg());
write('pipeline.svg', pipelineSvg());
write('hypothesis_mdl.svg', hypothesisSvg());
write('interventions.svg', interventionsSvg());
console.log('Wrote SVG figures to figures/.');
