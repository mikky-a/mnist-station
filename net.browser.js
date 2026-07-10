'use strict';
// Браузерная версия ядра сети — те же функции, что net.js, но глобальные.
// (В браузере нет require/module, поэтому просто объявляем в window.)

const sigmoid = z => 1 / (1 + Math.exp(-z));

function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeNet(sizes) {
  const W = [], B = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const inN = sizes[l], outN = sizes[l + 1];
    const w = [];
    for (let i = 0; i < outN; i++) {
      const row = new Float64Array(inN);
      for (let j = 0; j < inN; j++) row[j] = randn() / Math.sqrt(inN);
      w.push(row);
    }
    W.push(w);
    const b = new Float64Array(outN);
    for (let i = 0; i < outN; i++) b[i] = randn();
    B.push(b);
  }
  return { sizes, W, B };
}

function forward(net, input, collect) {
  let signal = input;
  const saved = collect ? { acts: [input], zs: [] } : null;
  for (let l = 0; l < net.W.length; l++) {
    const W = net.W[l], B = net.B[l], outN = W.length, inN = W[0].length;
    const z = new Float64Array(outN), a = new Float64Array(outN);
    for (let i = 0; i < outN; i++) {
      let s = 0;
      for (let j = 0; j < inN; j++) s += signal[j] * W[i][j];
      s += B[i];
      z[i] = s;
      a[i] = sigmoid(s);
    }
    signal = a;
    if (collect) { saved.zs.push(z); saved.acts.push(a); }
  }
  return collect ? saved : signal;
}

function backward(net, saved, y) {
  const a = saved.acts, L = net.W.length;
  const gradW = net.W.map(w => w.map(row => new Float64Array(row.length)));
  const gradB = net.B.map(b => new Float64Array(b.length));
  let last = a[a.length - 1];
  let delta = new Float64Array(last.length);
  for (let i = 0; i < last.length; i++)
    delta[i] = (last[i] - y[i]) * last[i] * (1 - last[i]);
  for (let l = L - 1; l >= 0; l--) {
    const aIn = a[l];
    for (let i = 0; i < delta.length; i++) {
      gradB[l][i] = delta[i];
      const row = gradW[l][i];
      for (let j = 0; j < aIn.length; j++) row[j] = delta[i] * aIn[j];
    }
    if (l > 0) {
      const W = net.W[l];
      const next = new Float64Array(aIn.length);
      for (let j = 0; j < aIn.length; j++) {
        let s = 0;
        for (let i = 0; i < delta.length; i++) s += W[i][j] * delta[i];
        next[j] = s * aIn[j] * (1 - aIn[j]);
      }
      delta = next;
    }
  }
  return { gradW, gradB };
}

function updateMiniBatch(net, batch, eta) {
  const sumW = net.W.map(w => w.map(r => new Float64Array(r.length)));
  const sumB = net.B.map(b => new Float64Array(b.length));
  let lossSum = 0;                                      // лосс батча — бесплатно, forward уже сделан
  for (const { x, y } of batch) {
    const saved = forward(net, x, true);
    const out = saved.acts[saved.acts.length - 1];
    let s = 0; for (let i = 0; i < out.length; i++) { const d = y[i] - out[i]; s += d * d; }
    lossSum += 0.5 * s;
    const { gradW, gradB } = backward(net, saved, y);
    for (let l = 0; l < net.W.length; l++)
      for (let i = 0; i < net.W[l].length; i++) {
        for (let j = 0; j < net.W[l][i].length; j++) sumW[l][i][j] += gradW[l][i][j];
        sumB[l][i] += gradB[l][i];
      }
  }
  const k = eta / batch.length;
  for (let l = 0; l < net.W.length; l++)
    for (let i = 0; i < net.W[l].length; i++) {
      for (let j = 0; j < net.W[l][i].length; j++) net.W[l][i][j] -= k * sumW[l][i][j];
      net.B[l][i] -= k * sumB[l][i];
    }
  return lossSum / batch.length;
}

function argmax(v) { let m = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[m]) m = i; return m; }

function evaluate(net, test) {
  let ok = 0;
  for (const s of test) if (argmax(forward(net, s.x, false)) === s.label) ok++;
  return ok;
}
