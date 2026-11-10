/**
 * Interactive Application Logic (web/app.js)
 * ES 335: Cost-Aware Credit Card Default Prediction
 * Anant Shukla (25110030) & Jay R. Glasswala (25110147)
 */

let globalData = null;
let metricsData = null;
let fairnessChart = null;
let calibrationChart = null;

// Group label decoders
const GROUP_LABELS = {
  'SEX': { 1: 'Male', 2: 'Female' },
  'EDUCATION': { 1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others' },
  'MARRIAGE': { 1: 'Married', 2: 'Single', 3: 'Others' }
};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [testRes, metricsRes] = await Promise.all([
      fetch('data/test_results.json').then(r => r.json()),
      fetch('data/metrics_summary.json').then(r => r.json())
    ]);

    globalData = testRes;
    metricsData = metricsRes;

    console.log(`[App] Loaded ${globalData.clients.length} test instances.`);

    // Initialize UI components
    setupEventListeners();
    populateModelComparisonTable();
    initCalibrationChart();
    updateCostSimulator();
    updateFairnessAudit();

  } catch (err) {
    console.error('[App] Failed to load data:', err);
    // Helpful reminder if user opens via file:// instead of localhost
    if (window.location.protocol === 'file:') {
      alert("Note: Browsers block fetch() on file:// URLs. Please view this through a static web server (e.g. 'python -m http.server' inside the web directory) or on GitHub Pages!");
    }
  }
});

function setupEventListeners() {
  const slider = document.getElementById('cost-slider');
  const modelSelect = document.getElementById('model-select');
  const groupSelect = document.getElementById('fairness-group-select');

  slider.addEventListener('input', () => {
    const r = parseFloat(slider.value);
    document.getElementById('ratio-display').textContent = `${r.toFixed(1)} : 1`;
    document.getElementById('cm-fn-cost').textContent = r.toFixed(1);
    updateCostSimulator();
    updateFairnessAudit();
  });

  modelSelect.addEventListener('change', () => {
    updateCostSimulator();
    updateFairnessAudit();
    updateCalibrationCurve();
  });

  groupSelect.addEventListener('change', () => {
    updateFairnessAudit();
  });
}

/**
 * Live Cost Simulator: Recalculates metrics over 5,993 test records in < 5 ms.
 */
function updateCostSimulator() {
  if (!globalData) return;

  const modelKey = document.getElementById('model-select').value;
  const r = parseFloat(document.getElementById('cost-slider').value);
  const tStar = 1.0 / (1.0 + r);

  document.getElementById('active-thresh').textContent = tStar.toFixed(3);

  let tn = 0, fp = 0, fn = 0, tp = 0;
  const clients = globalData.clients;
  const n = clients.length;

  for (let i = 0; i < n; i++) {
    const c = clients[i];
    const p = c.p[modelKey] !== undefined ? c.p[modelKey] : 0.0;
    const yPred = p >= tStar ? 1 : 0;
    const yTrue = c.y;

    if (yTrue === 0 && yPred === 0) tn++;
    else if (yTrue === 0 && yPred === 1) fp++;
    else if (yTrue === 1 && yPred === 0) fn++;
    else if (yTrue === 1 && yPred === 1) tp++;
  }

  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
  const costPerClient = (fn * r + fp) / n;

  // Live metrics display
  document.getElementById('active-recall').textContent = `${(recall * 100).toFixed(1)}%`;
  document.getElementById('active-fpr').textContent = `${(fpr * 100).toFixed(1)}%`;
  document.getElementById('active-cost').textContent = `${costPerClient.toFixed(3)} c_FP`;

  // Confusion matrix counts
  document.getElementById('cm-tn').textContent = tn.toLocaleString();
  document.getElementById('cm-fp').textContent = fp.toLocaleString();
  document.getElementById('cm-fn').textContent = fn.toLocaleString();
  document.getElementById('cm-tp').textContent = tp.toLocaleString();

  // Benchmark against trivial policies
  const baseRate = globalData.base_rate;
  const costFlagNobody = baseRate * r;
  const costFlagEverybody = (1.0 - baseRate) * 1.0;

  // T = 0.50 naive cost
  let fn50 = 0, fp50 = 0;
  for (let i = 0; i < n; i++) {
    const c = clients[i];
    const p = c.p[modelKey] !== undefined ? c.p[modelKey] : 0.0;
    const yPred = p >= 0.50 ? 1 : 0;
    if (c.y === 1 && yPred === 0) fn50++;
    if (c.y === 0 && yPred === 1) fp50++;
  }
  const costP50 = (fn50 * r + fp50) / n;

  document.getElementById('policy-model-cost').textContent = `${costPerClient.toFixed(3)} c_FP`;
  document.getElementById('policy-p50-cost').textContent = `${costP50.toFixed(3)} c_FP`;
  document.getElementById('policy-nobody-cost').textContent = `${costFlagNobody.toFixed(3)} c_FP`;
  document.getElementById('policy-everybody-cost').textContent = `${costFlagEverybody.toFixed(3)} c_FP`;
}

/**
 * Fairness Audit: Computes error disparities at active threshold t*.
 */
function updateFairnessAudit() {
  if (!globalData) return;

  const modelKey = document.getElementById('model-select').value;
  const attrKey = document.getElementById('fairness-group-select').value;
  const r = parseFloat(document.getElementById('cost-slider').value);
  const tStar = 1.0 / (1.0 + r);

  const attrCodeMap = { 'SEX': 's', 'EDUCATION': 'e', 'MARRIAGE': 'm' };
  const shortAttr = attrCodeMap[attrKey];

  const clients = globalData.clients;
  const groups = {};

  // Aggregate by group
  for (let i = 0; i < clients.length; i++) {
    const c = clients[i];
    const gVal = c[shortAttr];
    if (!groups[gVal]) {
      groups[gVal] = { total: 0, defaulters: 0, fp: 0, fn: 0, nonDefaulters: 0 };
    }

    const p = c.p[modelKey] !== undefined ? c.p[modelKey] : 0.0;
    const yPred = p >= tStar ? 1 : 0;
    const yTrue = c.y;

    groups[gVal].total++;
    if (yTrue === 1) {
      groups[gVal].defaulters++;
      if (yPred === 0) groups[gVal].fn++;
    } else {
      groups[gVal].nonDefaulters++;
      if (yPred === 1) groups[gVal].fp++;
    }
  }

  // Populate Table and Chart Data
  const labels = [];
  const farData = [];
  const mdrData = [];
  const tbody = document.getElementById('fairness-tbody');
  tbody.innerHTML = '';

  const sortedKeys = Object.keys(groups).sort((a, b) => parseInt(a) - parseInt(b));

  for (const g of sortedKeys) {
    const gData = groups[g];
    const gName = GROUP_LABELS[attrKey][g] || `Group ${g}`;
    const far = gData.nonDefaulters > 0 ? (gData.fp / gData.nonDefaulters) : 0;
    const mdr = gData.defaulters > 0 ? (gData.fn / gData.defaulters) : 0;
    const baseRate = gData.total > 0 ? (gData.defaulters / gData.total) : 0;
    const isSmall = gData.total < 100;

    labels.push(gName + (isSmall ? ' *' : ''));
    farData.push(parseFloat((far * 100).toFixed(1)));
    mdrData.push(parseFloat((mdr * 100).toFixed(1)));

    const tr = document.createElement('tr');
    if (isSmall) tr.style.opacity = '0.6';

    tr.innerHTML = `
      <td><strong>${gName}</strong> ${isSmall ? '<span class="badge badge-warning" style="font-size:0.65rem;">Small N</span>' : ''}</td>
      <td>${gData.total.toLocaleString()}</td>
      <td>${(baseRate * 100).toFixed(1)}%</td>
      <td><strong>${(far * 100).toFixed(1)}%</strong> (${gData.fp}/${gData.nonDefaulters})</td>
      <td><strong>${(mdr * 100).toFixed(1)}%</strong> (${gData.fn}/${gData.defaulters})</td>
    `;
    tbody.appendChild(tr);
  }

  // Update Chart.js
  renderFairnessChart(labels, farData, mdrData);
}

function renderFairnessChart(labels, farData, mdrData) {
  const ctx = document.getElementById('fairnessChart').getContext('2d');
  if (fairnessChart) {
    fairnessChart.destroy();
  }

  fairnessChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'False Alarm Rate (%)',
          data: farData,
          backgroundColor: 'rgba(245, 158, 11, 0.75)',
          borderColor: 'rgba(245, 158, 11, 1)',
          borderWidth: 1
        },
        {
          label: 'Missed Defaulter Rate (%)',
          data: mdrData,
          backgroundColor: 'rgba(239, 68, 68, 0.75)',
          borderColor: 'rgba(239, 68, 68, 1)',
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: 'Error Rate (%)' }
        }
      },
      plugins: {
        legend: { position: 'top' }
      }
    }
  });
}

/**
 * Populate Model Comparison Table from precomputed evaluation metrics
 */
function populateModelComparisonTable() {
  if (!metricsData || !metricsData.evaluation) return;

  const evalObj = metricsData.evaluation;
  const tbody = document.getElementById('results-tbody');
  tbody.innerHTML = '';

  const modelLabels = {
    'hist_gradient_boosting': 'HistGradientBoosting (Best)',
    'random_forest': 'Random Forest',
    'rbf_svm': 'RBF-Kernel SVM (Platt Scaled)',
    'ridge_logistic': 'Ridge Logistic (L2)',
    'lasso_logistic': 'Lasso Logistic (L1)',
    'elastic_net': 'Elastic-Net Logistic',
    'plain_logistic': 'Plain Logistic (Reference)',
    'majority_class': 'Majority Class (Baseline)'
  };

  for (const [key, label] of Object.entries(modelLabels)) {
    if (!evalObj[key]) continue;
    const m = evalObj[key];
    const ci = m.bootstrap_ci95;

    const tr = document.createElement('tr');
    if (key === 'hist_gradient_boosting') {
      tr.style.backgroundColor = '#f0fdf4';
      tr.style.fontWeight = 'bold';
    }

    const prAucCI = ci ? `[${ci.pr_auc_ci95[0]}, ${ci.pr_auc_ci95[1]}]` : '--';
    const llCI = ci ? `[${ci.log_loss_ci95[0]}, ${ci.log_loss_ci95[1]}]` : '--';
    const costCI = ci ? `[${ci.expected_cost_ci95[0]}, ${ci.expected_cost_ci95[1]}]` : '--';

    tr.innerHTML = `
      <td>${label}</td>
      <td><strong>${m.pr_auc.toFixed(4)}</strong> <span class="text-sm text-muted">${prAucCI}</span></td>
      <td>${m.log_loss.toFixed(4)} <span class="text-sm text-muted">${llCI}</span></td>
      <td>${m.brier_score.toFixed(4)}</td>
      <td>${m.ece_equal_frequency.toFixed(4)}</td>
      <td><strong>${m.cost_r5.toFixed(4)}</strong> <span class="text-sm text-muted">${costCI}</span></td>
      <td>${(m.recall_at_p50 * 100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  }

  // Paired test summary
  if (evalObj.paired_difference_test) {
    const pt = evalObj.paired_difference_test.delta_metrics;
    document.getElementById('delta-prauc').textContent = `+${pt.delta_pr_auc_ci95[0]} to +${pt.delta_pr_auc_ci95[1]} (Strictly Positive)`;
    document.getElementById('delta-cost').textContent = `${pt.delta_cost_ci95[0]} to ${pt.delta_cost_ci95[1]} c_FP (Cost Reduction)`;
  }
}

/**
 * Probability Calibration Reliability Curves
 */
function initCalibrationChart() {
  updateCalibrationCurve();
}

function updateCalibrationCurve() {
  if (!metricsData || !metricsData.evaluation) return;

  const modelKey = document.getElementById('model-select').value;
  const m = metricsData.evaluation[modelKey];
  if (!m || !m.reliability_curve) return;

  const curve = m.reliability_curve;
  const predVals = curve.map(b => b.mean_predicted);
  const empVals = curve.map(b => b.empirical_fraction);

  const ctx = document.getElementById('calibrationChart').getContext('2d');
  if (calibrationChart) {
    calibrationChart.destroy();
  }

  calibrationChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: predVals.map(v => v.toFixed(2)),
      datasets: [
        {
          label: `${modelKey} Calibration`,
          data: empVals,
          borderColor: '#1e3a8a',
          backgroundColor: '#3b82f6',
          borderWidth: 2,
          pointRadius: 4,
          fill: false,
          tension: 0.1
        },
        {
          label: 'Perfect Calibration (Ideal)',
          data: predVals,
          borderColor: '#94a3b8',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Mean Predicted Probability (Quantile)' } },
        y: { beginAtZero: true, max: 0.8, title: { display: true, text: 'Observed Empirical Default Rate' } }
      },
      plugins: {
        legend: { position: 'top' }
      }
    }
  });
}
