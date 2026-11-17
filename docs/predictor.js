// Prediction code for the "Check a client" section.
// It recreates our scikit-learn models from data/predictor.json:
// scale the numbers, one-hot encode education, then Ridge = logistic of a
// weighted sum, HGB = starting score + one leaf value from each tree.
// The same functions are used by tests/test_predictor_parity.py (through node)
// to check they match the Python probabilities for every test client.

function preprocess(model, client) {
  const pre = model.preprocessing;
  const x = pre.numeric.map((col, i) => (Number(client[col]) - pre.mean[i]) / pre.scale[i]);
  pre.education_columns.forEach(level => x.push(Number(client.EDUCATION) === level ? 1 : 0));
  return x;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictRidge(model, client) {
  const x = preprocess(model, client);
  let z = model.ridge.intercept;
  model.ridge.coef.forEach((w, i) => { z += w * x[i]; });
  return sigmoid(z);
}

function predictHGB(model, client) {
  const x = preprocess(model, client);
  let raw = model.hgb.baseline;
  for (const tree of model.hgb.trees) {
    let node = 0;
    while (!tree.leaf[node]) {
      const value = x[tree.f[node]];
      const goLeft = Number.isNaN(value) ? tree.m[node] === 1 : value <= tree.t[node];
      node = goLeft ? tree.l[node] : tree.r[node];
    }
    raw += tree.v[node];
  }
  return sigmoid(raw);
}

if (typeof module !== 'undefined') {
  module.exports = { preprocess, predictRidge, predictHGB };
}
