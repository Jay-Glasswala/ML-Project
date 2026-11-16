// Load data once, populate the model selector, and redraw every time the user picks a new model.
const dataPath = "data/";          // folder where JSON lives
const metricsFile = `${dataPath}evaluation_metrics.json`;
const fairnessFile = `${dataPath}fairness_audit_report.json`;

// Helper: create a Bootstrap card for the metric numbers
function makeMetricCard(label, value) {
  const div = document.createElement("div");
  div.className = "col-sm-3 metric-card";
  div.innerHTML = `
    <div class="label text-muted">${label}</div>
    <div class="value">${value}</div>`;
  return div;
}

Promise.all([
  fetch(metricsFile).then(r => r.json()),
  fetch(fairnessFile).then(r => r.json())
]).then(([metrics, fairness]) => {
  const models = Object.keys(metrics);
  const selector = document.getElementById("modelSelect");
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m.replace(/_/g, " ").toUpperCase();
    selector.appendChild(opt);
  });

  const render = () => {
    const model = selector.value;
    const m = metrics[model];
    const f = fairness[model];

    // ---- Metric cards ----
    const cardContainer = document.getElementById("metricCards");
    cardContainer.innerHTML = "";
    cardContainer.appendChild(makeMetricCard("PR‑AUC", m.pr_auc.toFixed(3)));
    cardContainer.appendChild(makeMetricCard("Log‑Loss", m.log_loss.toFixed(3)));
    cardContainer.appendChild(makeMetricCard("ECE", m.ece.toFixed(3)));
    cardContainer.appendChild(makeMetricCard("Cost (r=5)", m.cost.toFixed(3)));

    // ---- PR‑AUC bar ----
    Plotly.newPlot("chart-auc", [{
      x: [model],
      y: [m.pr_auc],
      type: "bar",
      marker: {color: "#0066CC"}
    }], {margin: {t:20, b:30}, yaxis: {title:"PR‑AUC"}});

    // ---- Log‑Loss bar ----
    Plotly.newPlot("chart-logloss", [{
      x: [model],
      y: [m.log_loss],
      type: "bar",
      marker: {color: "#FF6600"}
    }], {margin: {t:20, b:30}, yaxis: {title:"Log‑Loss"}});

    // ---- Cost bar ----
    Plotly.newPlot("chart-cost", [{
      x: [model],
      y: [m.cost],
      type: "bar",
      marker: {color: "#28A745"}
    }], {margin: {t:20, b:30}, yaxis: {title:"Cost (r=5)"}});

    // ---- Fairness heat‑map ----
    if (f) {
      const attrs = Object.keys(f);
      const groups = [];
      attrs.forEach(attr => {
        f[attr].forEach(g => {
          groups.push({attr, group: g.group, FAR: g.FAR, MDR: g.MDR});
        });
      });
      const df = groups.reduce((acc, row) => {
        acc.attr.push(row.attr);
        acc.group.push(row.group);
        acc.FAR.push(row.FAR);
        acc.MDR.push(row.MDR);
        return acc;
      }, {attr: [], group: [], FAR: [], MDR: []});

      const traceFAR = {
        x: df.group,
        y: df.attr,
        z: df.FAR,
        type: "heatmap",
        colorscale: "Blues",
        name: "FAR",
        hovertemplate: "FAR: %{z:.3f}<extra></extra>"
      };
      const traceMDR = {
        x: df.group,
        y: df.attr,
        z: df.MDR,
        type: "heatmap",
        colorscale: "Oranges",
        name: "MDR",
        hovertemplate: "MDR: %{z:.3f}<extra></extra>"
      };
      Plotly.newPlot("chart-fairness", [traceFAR, traceMDR], {
        title: "Fairness (group‑wise)",
        grid: {rows: 1, columns: 2, pattern: "independent"},
        margin: {t:30}
      });
    } else {
      document.getElementById("chart-fairness").innerHTML = "<p>No fairness data for this model.</p>";
    }
  };

  selector.addEventListener("change", render);
  selector.value = models[0];
  render();
});
