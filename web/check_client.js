// "Check a client" section (7b). The model maths is in predictor.js;
// this file only handles the form, the example buttons and the output text.

const CC_FIELDS = [
  'LIMIT_BAL', 'AGE', 'EDUCATION',
  'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
];

const CC_NAMES = {
  LIMIT_BAL: 'Credit limit', AGE: 'Age', EDUCATION: 'Education',
  PAY_1: 'Sep repayment status', PAY_2: 'Aug repayment status', PAY_3: 'Jul repayment status',
  PAY_4: 'Jun repayment status', PAY_5: 'May repayment status', PAY_6: 'Apr repayment status',
  BILL_AMT1: 'Sep bill', BILL_AMT2: 'Aug bill', BILL_AMT3: 'Jul bill',
  BILL_AMT4: 'Jun bill', BILL_AMT5: 'May bill', BILL_AMT6: 'Apr bill',
  PAY_AMT1: 'Sep payment', PAY_AMT2: 'Aug payment', PAY_AMT3: 'Jul payment',
  PAY_AMT4: 'Jun payment', PAY_AMT5: 'May payment', PAY_AMT6: 'Apr payment'
};

let ccModel = null;
let ccTestClients = null;   // all test clients, loaded for the random button
let ccSource = null;        // where the current form values came from: {title, actual}

document.addEventListener('DOMContentLoaded', async () => {
  try {
    ccModel = await fetch('data/predictor.json').then(r => r.json());
  } catch (err) {
    console.error('Could not load predictor.json', err);
    document.getElementById('cc-decision').textContent = 'Could not load the model file (data/predictor.json).';
    return;
  }

  // dropdown of client types, each a real test client
  const select = document.getElementById('cc-example-select');
  ccModel.examples.forEach((ex, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = ex.label;
    select.appendChild(opt);
  });
  select.addEventListener('change', () => loadExample(Number(select.value)));

  document.getElementById('cc-random-btn').addEventListener('click', loadRandomClient);

  document.querySelectorAll('.cc-input').forEach(el => {
    el.addEventListener('input', () => { clearExampleHighlight(); updateClientPrediction(); });
  });
  const slider = document.getElementById('cost-slider');
  if (slider) slider.addEventListener('input', updateClientPrediction);

  loadExample(0);
});

function fillForm(inputs) {
  CC_FIELDS.forEach(f => { document.getElementById('cc-' + f).value = inputs[f]; });
}

function loadExample(i) {
  const ex = ccModel.examples[i];
  fillForm(ex.inputs);
  document.getElementById('cc-example-select').value = i;
  document.getElementById('cc-example-desc').textContent = ex.n_matching === null
    ? ex.description
    : `${ex.description} In the test set, ${ex.n_matching.toLocaleString()} clients match this type and ` +
      `${(ex.default_rate_matching * 100).toFixed(1)}% of them defaulted. We loaded the most typical one.`;
  ccSource = { title: `"${ex.label}" example (test client #${ex.test_index})`, actual: ex.actual_default };
  updateClientPrediction();
}

async function loadRandomClient() {
  const btn = document.getElementById('cc-random-btn');
  if (!ccTestClients) {
    btn.textContent = 'Loading...';
    try {
      ccTestClients = await fetch('data/test_clients.json').then(r => r.json());
    } catch (err) {
      console.error('Could not load test_clients.json', err);
      btn.textContent = 'Random test client';
      return;
    }
    btn.textContent = 'Random test client';
  }

  const filter = document.getElementById('cc-random-filter').value;
  const t = currentThreshold();
  const candidates = [];
  ccTestClients.rows.forEach((row, i) => {
    const y = ccTestClients.default[i];
    if (filter === 'default' && y !== 1) return;
    if (filter === 'no_default' && y !== 0) return;
    if (filter === 'mistake') {
      const predicted = predictHGB(ccModel, rowToClient(row)) >= t ? 1 : 0;
      if (predicted === y) return;
    }
    candidates.push(i);
  });
  if (candidates.length === 0) return;

  const i = candidates[Math.floor(Math.random() * candidates.length)];
  fillForm(rowToClient(ccTestClients.rows[i]));
  document.getElementById('cc-example-select').selectedIndex = -1;
  const groupText = {
    any: 'test clients',
    default: 'test clients who defaulted',
    no_default: 'test clients who did not default',
    mistake: 'test clients the model gets wrong at the current cost ratio'
  }[filter];
  document.getElementById('cc-example-desc').textContent =
    `Picked at random from the ${candidates.length.toLocaleString()} ${groupText}. Click again for another one.`;
  ccSource = { title: `Random test client #${i}`, actual: ccTestClients.default[i] };
  updateClientPrediction();
}

function rowToClient(row) {
  const client = {};
  ccTestClients.columns.forEach((c, j) => { client[c] = row[j]; });
  return client;
}

function currentThreshold() {
  const slider = document.getElementById('cost-slider');
  const r = slider ? parseFloat(slider.value) : 5;
  return 1 / (1 + r);
}

function clearExampleHighlight() {
  if (!ccSource) return;
  ccSource = null;
  document.getElementById('cc-example-select').selectedIndex = -1;
  document.getElementById('cc-example-desc').textContent = 'You edited the inputs, so this is now your own client.';
  document.getElementById('cc-example-note').textContent = '';
}

function readClient() {
  const client = {};
  for (const f of CC_FIELDS) {
    const raw = document.getElementById('cc-' + f).value;
    if (raw === '' || Number.isNaN(Number(raw))) return null;
    client[f] = Number(raw);
  }
  return client;
}

function typicalValue(field) {
  if (field.startsWith('PAY_') && !field.startsWith('PAY_AMT')) return 0;  // "paid minimum"
  if (field === 'EDUCATION') return 2;  // most common level (university)
  return ccModel.train_medians[field];
}

function updateClientPrediction() {
  if (!ccModel) return;
  const client = readClient();
  const decision = document.getElementById('cc-decision');
  if (!client) {
    decision.textContent = 'Fill in every field to get a prediction.';
    ['cc-p-hgb', 'cc-p-ridge'].forEach(id => { document.getElementById(id).textContent = '--'; });
    return;
  }

  const pHGB = predictHGB(ccModel, client);
  const pRidge = predictRidge(ccModel, client);
  const slider = document.getElementById('cost-slider');
  const r = slider ? parseFloat(slider.value) : 5;
  const t = 1 / (1 + r);

  document.getElementById('cc-p-hgb').textContent = pHGB.toFixed(3);
  document.getElementById('cc-p-ridge').textContent = pRidge.toFixed(3);
  document.getElementById('cc-r').textContent = r.toFixed(1);
  document.getElementById('cc-t').textContent = t.toFixed(3);

  decision.innerHTML = pHGB >= t
    ? `<div class="cc-verdict"><span class="badge-danger">Prediction: will default</span></div>
       HGB probability ${pHGB.toFixed(3)} &ge; threshold ${t.toFixed(3)}. At a cost ratio of ${r.toFixed(1)} : 1, missing this client would cost more than a false alarm, so the bank should review them.`
    : `<div class="cc-verdict"><span class="badge-success">Prediction: won't default</span></div>
       HGB probability ${pHGB.toFixed(3)} &lt; threshold ${t.toFixed(3)}. At a cost ratio of ${r.toFixed(1)} : 1, flagging this client would most likely be a false alarm.`;

  // what actually happened, if the client is a real test client
  const note = document.getElementById('cc-example-note');
  if (ccSource) {
    const predicted = pHGB >= t ? 1 : 0;
    note.innerHTML = `<div class="cc-outcome"><strong>${ccSource.title}.</strong> What actually happened: ` +
      `${ccSource.actual ? 'defaulted' : 'did not default'} in October 2005, so at this cost ratio the model is ` +
      (predicted === ccSource.actual ? '<span class="badge-success">right</span>' : '<span class="badge-danger">wrong</span>') +
      '.</div>';
  } else {
    note.textContent = '';
  }

  // what-if: replace one input at a time with a typical value
  const effects = CC_FIELDS.map(f => {
    const changed = Object.assign({}, client, { [f]: typicalValue(f) });
    return { field: f, from: client[f], to: typicalValue(f), p: predictHGB(ccModel, changed) };
  }).filter(e => e.from !== e.to)
    .sort((a, b) => Math.abs(b.p - pHGB) - Math.abs(a.p - pHGB))
    .slice(0, 3)
    .filter(e => Math.abs(e.p - pHGB) >= 0.005);

  const list = document.getElementById('cc-whatif');
  list.innerHTML = effects.length === 0
    ? '<li>No single input moves the probability by more than 0.005. This client already looks typical.</li>'
    : effects.map(e => `<li><strong>${CC_NAMES[e.field]}</strong> (${describe(e.field, e.from)} &rarr; ${describe(e.field, e.to)}): probability ${pHGB.toFixed(3)} &rarr; ${e.p.toFixed(3)}</li>`).join('');

  // warn when inputs are outside what the model saw in training
  const outside = Object.entries(ccModel.input_ranges)
    .filter(([f, [lo, hi]]) => client[f] < lo || client[f] > hi)
    .map(([f]) => CC_NAMES[f]);
  document.getElementById('cc-warning').textContent = outside.length
    ? `Outside the range seen in training: ${outside.join(', ')}. The prediction for this client is less reliable.`
    : '';
}

function describe(field, value) {
  if (field.startsWith('PAY_') && !field.startsWith('PAY_AMT')) {
    if (value === -2) return 'no bill';
    if (value === -1) return 'paid in full';
    if (value === 0) return 'paid minimum';
    return `${value} month${value > 1 ? 's' : ''} late`;
  }
  if (field === 'EDUCATION') return ['', 'graduate school', 'university', 'high school', 'others'][value] || value;
  if (field === 'AGE') return value;
  return Math.round(value).toLocaleString();
}
