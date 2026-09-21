"use strict";

const $ = (selector) => document.querySelector(selector);
const benchmarkState = {
  payload: null,
  target: "artifact",
  metric: "balanced_accuracy",
  model: "all",
  branches: "all",
  selectedExperiment: null,
};

document.addEventListener("DOMContentLoaded", loadBenchmark);

async function loadBenchmark() {
  try {
    const response = await fetch("/benchmark_results.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Result file returned ${response.status}`);
    benchmarkState.payload = await response.json();
    bindBenchmarkControls();
    renderStaticSections();
    renderResults();
  } catch (error) {
    $("#benchmarkLoading").innerHTML = `<strong>Benchmark results could not be loaded</strong><span>${escapeHtml(error.message)}</span>`;
    return;
  }
  $("#benchmarkLoading").hidden = true;
}

function bindBenchmarkControls() {
  const { targets, models, combinations } = benchmarkState.payload;
  fillSelect($("#targetSelect"), targets.map((item) => [item.id, item.label]));
  fillSelect($("#modelSelect"), models.map((item) => [item.id, item.label]), true);
  fillSelect($("#branchSelect"), combinations.map((item) => [item.id, `${item.id} · ${item.feature_count} features`]), true);
  $("#targetSelect").value = benchmarkState.target;
  $("#targetSelect").addEventListener("change", (event) => setTarget(event.target.value));
  $("#metricSelect").addEventListener("change", (event) => { benchmarkState.metric = event.target.value; renderResults(); });
  $("#modelSelect").addEventListener("change", (event) => { benchmarkState.model = event.target.value; renderResults(); });
  $("#branchSelect").addEventListener("change", (event) => { benchmarkState.branches = event.target.value; renderResults(); });
  $("#resetFilters").addEventListener("click", () => {
    benchmarkState.metric = "balanced_accuracy";
    benchmarkState.model = "all";
    benchmarkState.branches = "all";
    $("#metricSelect").value = benchmarkState.metric;
    $("#modelSelect").value = benchmarkState.model;
    $("#branchSelect").value = benchmarkState.branches;
    renderResults();
  });
}

function fillSelect(select, entries, preserveFirst = false) {
  const first = preserveFirst ? select.firstElementChild : null;
  select.replaceChildren();
  if (first) select.append(first);
  for (const [value, label] of entries) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.append(option);
  }
}

function setTarget(target) {
  benchmarkState.target = target;
  benchmarkState.selectedExperiment = null;
  $("#targetSelect").value = target;
  document.querySelectorAll(".target-card").forEach((card) => card.classList.toggle("active", card.dataset.target === target));
  renderResults();
}

function renderStaticSections() {
  const payload = benchmarkState.payload;
  $("#summaryRows").textContent = integer(payload.dataset.rows);
  $("#summaryFeatures").textContent = integer(payload.dataset.feature_columns);
  $("#summaryExperiments").textContent = integer(payload.results.length);
  $("#summaryDatasets").textContent = integer(payload.dataset.datasets.length);
  $("#summaryQa").textContent = payload.dataset.validation_passed ? "PASS" : "CHECK";

  const targetGrid = $("#targetGrid");
  for (const target of payload.targets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `target-card${target.id === benchmarkState.target ? " active" : ""}`;
    button.dataset.target = target.id;
    button.innerHTML = `<span>${escapeHtml(target.label)}</span><strong>${integer(target.class_0)} <i>0</i> / ${integer(target.class_1)} <i>1</i></strong><p>${escapeHtml(target.definition)}</p><small>${target.test_groups} held-out lineage group${target.test_groups === 1 ? "" : "s"}</small>`;
    button.addEventListener("click", () => setTarget(target.id));
    targetGrid.append(button);
  }

  const modelCards = $("#modelCards");
  for (const [index, model] of payload.models.entries()) {
    const values = payload.results.filter((result) => result.model_id === model.id).map((result) => result.balanced_accuracy);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const card = document.createElement("article");
    card.className = "benchmark-model-card";
    card.innerHTML = `<div><i>${String(index + 1).padStart(2, "0")}</i><span>${escapeHtml(model.family)}</span></div><h3>${escapeHtml(model.label)}</h3><p>${escapeHtml(model.rationale)}</p><dl><div><dt>Mean balanced accuracy</dt><dd>${percent(mean)}</dd></div><div><dt>Frozen configuration</dt><dd>${escapeHtml(configText(model.configuration))}</dd></div></dl><a href="${escapeAttribute(model.reference_url)}" target="_blank" rel="noreferrer">${escapeHtml(model.reference_title)} ↗</a>`;
    modelCards.append(card);
  }

  const datasetNames = {
    butqdb: "BUT-QDB quality",
    ludb: "LUDB morphology",
    mitdb: "MIT-BIH arrhythmia",
    nstdb: "MIT-BIH NSTDB",
    qtdb: "QT Database",
    seizure_edf: "CHB-MIT + Siena ECG",
  };
  for (const dataset of payload.dataset.datasets) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${escapeHtml(datasetNames[dataset.key] || dataset.key)}</td><td>${integer(dataset.records)}</td><td>${integer(dataset.rows)}</td>`;
    $("#datasetTableBody").append(row);
  }
  for (const target of payload.targets) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${escapeHtml(target.label)}</td><td>${integer(target.class_0)}</td><td>${integer(target.class_1)}</td><td>${integer(target.train_rows)}</td><td>${integer(target.test_rows)}</td><td>${integer(target.train_groups)}</td><td>${integer(target.test_groups)}</td><td>${target.test_fold}</td>`;
    $("#splitTableBody").append(row);
  }
  $("#protocolHash").textContent = `Frozen source matrix SHA-256: ${payload.dataset.observations_sha256}`;
}

function renderResults() {
  const payload = benchmarkState.payload;
  const targetRows = payload.results.filter((result) => result.target_id === benchmarkState.target);
  let filtered = targetRows.filter((result) =>
    (benchmarkState.model === "all" || result.model_id === benchmarkState.model)
    && (benchmarkState.branches === "all" || result.combination_id === benchmarkState.branches)
  );
  filtered = [...filtered].sort((a, b) => b[benchmarkState.metric] - a[benchmarkState.metric] || b.balanced_accuracy - a.balanced_accuracy);
  if (!filtered.length) return;
  const best = filtered[0];
  renderBest(best);
  renderHeatmap(targetRows);
  renderResultTable(filtered);
  const selected = filtered.find((result) => result.experiment_id === benchmarkState.selectedExperiment) || best;
  renderExperimentDetail(selected);
}

function renderBest(result) {
  const target = benchmarkState.payload.targets.find((item) => item.id === result.target_id);
  $("#bestExperiment").querySelector("h3").textContent = `${result.model_label} · ${result.combination_id}`;
  $("#bestExperiment").querySelector("p").textContent = `${target.label} · ${result.feature_count} features · ${integer(result.test_rows)} held-out rows in ${result.test_groups} lineage group${result.test_groups === 1 ? "" : "s"}`;
  $("#bestExperiment").querySelector(".best-score strong").textContent = percent(result[benchmarkState.metric]);
  $("#bestExperiment").querySelector(".best-score small").textContent = metricLabel(benchmarkState.metric);
}

function renderHeatmap(targetRows) {
  const payload = benchmarkState.payload;
  const models = payload.models.filter((model) => benchmarkState.model === "all" || model.id === benchmarkState.model);
  const combinations = payload.combinations.filter((combo) => benchmarkState.branches === "all" || combo.id === benchmarkState.branches);
  const shell = $("#benchmarkHeatmap");
  shell.replaceChildren();
  shell.style.gridTemplateColumns = `170px repeat(${combinations.length}, minmax(72px, 1fr))`;
  shell.append(cell("", "heat-corner"));
  for (const combo of combinations) shell.append(cell(combo.id.replaceAll("+", " + "), "heat-column"));
  for (const model of models) {
    shell.append(cell(model.label, "heat-row"));
    for (const combo of combinations) {
      const result = targetRows.find((row) => row.model_id === model.id && row.combination_id === combo.id);
      const button = cell(percent(result[benchmarkState.metric]), "heat-cell", "button");
      button.type = "button";
      button.style.setProperty("--heat", heatColor(result[benchmarkState.metric]));
      button.title = `${model.label} · ${combo.id} · ${metricLabel(benchmarkState.metric)} ${percent(result[benchmarkState.metric])}`;
      button.classList.toggle("selected", result.experiment_id === benchmarkState.selectedExperiment);
      button.addEventListener("click", () => { benchmarkState.selectedExperiment = result.experiment_id; renderExperimentDetail(result); renderHeatmap(targetRows); });
      shell.append(button);
    }
  }
}

function cell(text, className, tag = "div") {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function renderResultTable(rows) {
  const body = $("#resultTableBody");
  body.replaceChildren();
  rows.forEach((result, index) => {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.classList.toggle("selected", result.experiment_id === benchmarkState.selectedExperiment);
    row.innerHTML = `<td>${index + 1}</td><td><b>${escapeHtml(result.combination_id)}</b></td><td>${escapeHtml(result.model_label)}</td><td>${result.feature_count}</td><td><strong>${percent(result.balanced_accuracy)}</strong></td><td>${percent(result.roc_auc)}</td><td>${percent(result.average_precision)}</td><td>${percent(result.f1)}</td><td>${percent(result.recall_sensitivity)}</td><td>${percent(result.specificity)}</td><td>${integer(result.test_rows)}</td>`;
    const select = () => { benchmarkState.selectedExperiment = result.experiment_id; renderExperimentDetail(result); document.querySelectorAll("#resultTableBody tr").forEach((item) => item.classList.toggle("selected", item === row)); renderHeatmap(benchmarkState.payload.results.filter((item) => item.target_id === benchmarkState.target)); };
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") select(); });
    body.append(row);
  });
  $("#resultCount").textContent = `${rows.length} experiment${rows.length === 1 ? "" : "s"}`;
}

function renderExperimentDetail(result) {
  benchmarkState.selectedExperiment = result.experiment_id;
  const target = benchmarkState.payload.targets.find((item) => item.id === result.target_id);
  $("#detailTitle").textContent = `${target.label} · ${result.model_label} · ${result.combination_id}`;
  $("#detailBalanced").textContent = percent(result.balanced_accuracy);
  const metrics = [
    ["ROC AUC", result.roc_auc], ["Average precision", result.average_precision],
    ["F1", result.f1], ["Sensitivity", result.recall_sensitivity],
    ["Specificity", result.specificity], ["Accuracy", result.accuracy],
  ];
  $("#metricGrid").replaceChildren(...metrics.map(([label, value]) => {
    const box = document.createElement("div"); box.innerHTML = `<span>${label}</span><strong>${percent(value)}</strong>`; return box;
  }));
  const confusion = $("#confusionGrid");
  confusion.replaceChildren();
  [["TN", result.tn, "actual 0 · predicted 0"], ["FP", result.fp, "actual 0 · predicted 1"], ["FN", result.fn, "actual 1 · predicted 0"], ["TP", result.tp, "actual 1 · predicted 1"]].forEach(([label, value, caption]) => {
    const box = document.createElement("div"); box.className = label === "TP" || label === "TN" ? "correct" : "error"; box.innerHTML = `<span>${label}</span><strong>${integer(value)}</strong><small>${caption}</small>`; confusion.append(box);
  });
  $("#detailNote").textContent = `${integer(result.train_rows)} training rows / ${integer(result.test_rows)} held-out rows; ${result.train_groups} training groups / ${result.test_groups} test group${result.test_groups === 1 ? "" : "s"}. ${result.complete_case_rows === 0 ? "No row had every selected feature; training-fold imputation retained explicit missingness indicators." : `${integer(result.complete_case_rows)} task rows had all ${result.feature_count} selected features present.`}`;
}

function metricLabel(key) {
  return ({ balanced_accuracy: "balanced accuracy", roc_auc: "ROC AUC", average_precision: "average precision", f1: "F1", recall_sensitivity: "sensitivity", specificity: "specificity" })[key] || key;
}

function heatColor(value) {
  const normalized = Math.max(0, Math.min(1, (Number(value) - 0.45) / 0.55));
  const hue = 8 + normalized * 160;
  const light = 94 - normalized * 50;
  return `hsl(${hue} 58% ${light}%)`;
}

function percent(value) { return value == null || !Number.isFinite(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(1)}%`; }
function integer(value) { return Number(value).toLocaleString("en-US"); }
function configText(configuration) { return Object.entries(configuration).map(([key, value]) => `${key}=${value}`).join(" · "); }
function escapeHtml(value) { const div = document.createElement("div"); div.textContent = String(value); return div.innerHTML; }
function escapeAttribute(value) { return String(value).replaceAll('"', "&quot;"); }
