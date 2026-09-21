"use strict";

const auditState = { payload: null, target: "artifact" };
const byId = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const response = await fetch("/audit_results.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Audit evidence returned ${response.status}`);
    auditState.payload = await response.json();
    renderStaticAudit();
    renderTargetAudit();
    byId("auditLoading").hidden = true;
  } catch (error) {
    byId("auditLoading").innerHTML = `<strong>Audit evidence could not be loaded</strong><span>${escapeHtml(error.message)}</span>`;
  }
});

function renderStaticAudit() {
  const p = auditState.payload;
  byId("auditVerdict").textContent = p.verdict;
  byId("rawRead").textContent = `${integer(p.raw_inventory.files_byte_read)} / ${integer(p.raw_inventory.files_planned)} files`;
  byId("rawBytes").textContent = `${bytes(p.raw_inventory.bytes_byte_read)} hashed; ${p.raw_inventory.read_errors.length} read errors`;
  byId("auditRows").textContent = integer(p.dataset.rows);
  byId("auditGroups").textContent = integer(p.dataset.lineage_groups);
  byId("auditExperiments").textContent = integer(p.results.length);
  byId("auditFiles").textContent = integer(p.raw_inventory.files_byte_read);
  const integrityPass = p.integrity.row_id_unique && p.integrity.source_coordinate_duplicate_rows === 0 && p.integrity.lineage_groups_crossing_suggested_folds === 0 && p.integrity.feature_names_with_identifier_or_label_tokens.length === 0 && (!p.label_semantics || p.label_semantics.all_passed) && (!p.independent_rebuild || p.independent_rebuild.exact_reproduction);
  byId("auditIntegrity").textContent = integrityPass ? "PASS" : "CHECK";

  const grid = byId("auditTargetGrid");
  p.targets.forEach((target) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `target-card${target.id === auditState.target ? " active" : ""}`;
    button.dataset.target = target.id;
    button.innerHTML = `<span>${escapeHtml(target.label)}</span><strong>${integer(target.class_0)} <i>0</i> / ${integer(target.class_1)} <i>1</i></strong><p>${escapeHtml(target.definition)}</p><small>${target.groups} independent group${target.groups === 1 ? "" : "s"} · ${escapeHtml(target.reliability)}</small>`;
    button.addEventListener("click", () => {
      auditState.target = target.id;
      document.querySelectorAll("#auditTargetGrid .target-card").forEach((card) => card.classList.toggle("active", card.dataset.target === target.id));
      renderTargetAudit();
    });
    grid.append(button);
  });

  const exhaustive = p.exhaustive_plan ? ` The separate raw exhaustive plan covers ${integer(p.exhaustive_plan.records_by_dataset ? Object.values(p.exhaustive_plan.records_by_dataset).reduce((sum, value) => sum + Number(value), 0) : 0)} records, ${integer(p.exhaustive_plan.planned_segments)} gap-free windows and ${Number(p.exhaustive_plan.signal_hours).toFixed(2)} signal-hours; unlabelled windows are not invented as negatives.` : "";
  const rebuilt = p.independent_rebuild ? ` Independent raw re-extraction exact match: ${p.independent_rebuild.exact_reproduction ? "yes" : "NO—review differences"}.` : "";
  byId("auditProtocol").innerHTML = `<strong>Exact protocol</strong><p>${escapeHtml(p.protocol.split)}. ${escapeHtml(p.protocol.preprocessing)}. ${escapeHtml(p.protocol.uncertainty)}. ${escapeHtml(p.protocol.selection_warning)}${escapeHtml(exhaustive)}${escapeHtml(rebuilt)}</p><code>${escapeHtml(p.dataset.matrix_sha256)}</code>`;
}

function renderTargetAudit() {
  const p = auditState.payload;
  const target = p.targets.find((item) => item.id === auditState.target);
  const selected = p.selected.find((item) => item.target_id === auditState.target);
  const permutation = p.permutations.find((item) => item.target_id === auditState.target);
  const baselines = p.baselines.filter((item) => item.target_id === auditState.target);

  byId("selectedReliability").textContent = target.reliability;
  byId("selectedTitle").textContent = `${selected.model_label} · ${selected.combination_id}`;
  byId("selectedScope").textContent = `${integer(selected.rows)} rows · ${selected.groups} lineage groups · ${selected.oof_folds} out-of-fold splits · ${selected.feature_count} features`;
  byId("selectedScore").textContent = percent(selected.group_macro_balanced_accuracy);
  byId("selectedCi").textContent = `Row-pooled BA 95% group bootstrap: ${percent(selected.balanced_accuracy_group_bootstrap_ci_low)}–${percent(selected.balanced_accuracy_group_bootstrap_ci_high)}`;

  const metrics = [
    ["Row balanced accuracy", selected.balanced_accuracy], ["Accuracy", selected.accuracy],
    ["Sensitivity", selected.recall_sensitivity], ["Specificity", selected.specificity],
    ["Precision", selected.precision], ["NPV", selected.npv],
    ["ROC AUC", selected.roc_auc], ["Average precision", selected.average_precision],
  ];
  byId("auditMetricGrid").replaceChildren(...metrics.map(([label, value]) => {
    const box = document.createElement("div"); box.innerHTML = `<span>${escapeHtml(label)}</span><strong>${percent(value)}</strong>`; return box;
  }));
  const cells = [["TN", selected.tn, "actual 0 · predicted 0"], ["FP", selected.fp, "actual 0 · predicted 1"], ["FN", selected.fn, "actual 1 · predicted 0"], ["TP", selected.tp, "actual 1 · predicted 1"]];
  byId("auditConfusion").replaceChildren(...cells.map(([label, value, note]) => {
    const box = document.createElement("div"); box.className = label === "TN" || label === "TP" ? "correct" : "error"; box.innerHTML = `<span>${label}</span><strong>${integer(value)}</strong><small>${escapeHtml(note)}</small>`; return box;
  }));
  byId("calculationText").textContent = `½ × [${selected.tp}/(${selected.tp}+${selected.fn}) + ${selected.tn}/(${selected.tn}+${selected.fp})] = ${percent(selected.balanced_accuracy)}`;

  const checks = [
    ["Missingness-only", baselines.find((item) => item.baseline === "feature missingness only")?.balanced_accuracy, "Can feature availability alone predict the target?"],
    ["Dataset-only", baselines.find((item) => item.baseline === "dataset identity only")?.balanced_accuracy, "Can source identity alone predict the target?"],
    ["Permuted labels", permutation.mean_balanced_accuracy, `${permutation.repetitions} repetitions; expected near chance`],
    ["Raw integrity", p.integrity.source_coordinate_duplicate_rows === 0 ? 1 : 0, `${p.integrity.source_coordinate_duplicate_rows} repeated source-time rows; ${p.integrity.exact_duplicate_feature_sets_crossing_groups} feature-vector sets cross groups`],
  ];
  byId("auditChecks").replaceChildren(...checks.map(([label, value, note], index) => {
    const card = document.createElement("article");
    const display = index === 3 ? (value === 1 ? "PASS" : "CHECK") : percent(value);
    const concerning = index < 3 && Number(value) >= 0.7;
    card.className = concerning ? "concerning" : "";
    card.innerHTML = `<span>${escapeHtml(label)}</span><strong>${display}</strong><p>${escapeHtml(note)}</p>`;
    return card;
  }));

  renderResults(selected.experiment_id);
  renderRecords();
  renderPatients();
  renderWeakCases();
  renderDatasets();
}

function renderResults(selectedId) {
  const rows = auditState.payload.results.filter((row) => row.target_id === auditState.target).sort((a, b) => a.rank_within_target - b.rank_within_target || b.balanced_accuracy - a.balanced_accuracy);
  const body = byId("auditResultsBody"); body.replaceChildren();
  rows.forEach((row) => {
    const tr = document.createElement("tr"); if (row.experiment_id === selectedId) tr.className = "selected";
    tr.innerHTML = `<td>${row.rank_within_target}</td><td><b>${escapeHtml(row.combination_id)}</b></td><td>${escapeHtml(row.model_label)}</td><td>${row.feature_count}</td><td><strong>${percent(row.group_macro_balanced_accuracy)}</strong></td><td>${percent(row.balanced_accuracy)}</td><td>${percent(row.balanced_accuracy_group_bootstrap_ci_low)}–${percent(row.balanced_accuracy_group_bootstrap_ci_high)}</td><td>${percent(row.recall_sensitivity)}</td><td>${percent(row.specificity)}</td><td>${integer(row.errors)}</td>`;
    body.append(tr);
  });
}

function renderRecords() {
  const rows = auditState.payload.records.filter((row) => row.target_id === auditState.target).sort((a, b) => (metricOrAccuracy(a) - metricOrAccuracy(b)) || b.errors - a.errors);
  const body = byId("recordBody"); body.replaceChildren();
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(row.dataset_key)}</td><td>${escapeHtml(row.record_id)}</td><td>${escapeHtml(row.lineage_group_id)}</td><td>${integer(row.rows)}</td><td>${integer(row.class_1)}</td><td>${integer(row.tn)}</td><td>${integer(row.fp)}</td><td>${integer(row.fn)}</td><td>${integer(row.tp)}</td><td>${percent(row.balanced_accuracy)}</td><td>${percent(row.errors / row.rows)}</td>`;
    body.append(tr);
  });
}

function renderPatients() {
  const rows = auditState.payload.patients.filter((row) => row.target_id === auditState.target).sort((a, b) => (metricOrAccuracy(a) - metricOrAccuracy(b)) || b.errors - a.errors);
  const body = byId("patientBody"); body.replaceChildren();
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(row.subject_id)}</td><td>${escapeHtml(row.lineage_group_id)}</td><td>${integer(row.rows)}</td><td>${integer(row.class_1)}</td><td>${integer(row.tn)}</td><td>${integer(row.fp)}</td><td>${integer(row.fn)}</td><td>${integer(row.tp)}</td><td>${percent(row.balanced_accuracy)}</td><td>${percent(row.errors / row.rows)}</td>`;
    body.append(tr);
  });
}

function renderWeakCases() {
  const rows = auditState.payload.weak_cases.filter((row) => row.target_id === auditState.target);
  const body = byId("weakCaseBody"); body.replaceChildren();
  rows.forEach((row) => {
    const label = row.label_beat_symbol_original ?? row.label_quality_consensus_original ?? row.label_seizure_phase_derived ?? row.label_noise_condition_original ?? "—";
    let resemblance = "—";
    try { resemblance = JSON.parse(row.descriptive_resemblance_json || "[]").slice(0, 3).map((item) => item.feature).join(", ") || "—"; } catch (_) { resemblance = "—"; }
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><b>${escapeHtml(row.error_type)}</b></td><td>${escapeHtml(row.dataset_key)}</td><td>${escapeHtml(row.record_id)}</td><td>${escapeHtml(row.lineage_group_id)}</td><td>${seconds(row.observation_time_s)}</td><td>${row.y_true}</td><td>${row.y_pred}</td><td>${number(row.decision_score)}</td><td>${escapeHtml(label)}</td><td>${escapeHtml(resemblance)}</td><td>${row.available_feature_count}/52</td>`;
    body.append(tr);
  });
}

function renderDatasets() {
  const targetKey = `${auditState.target}_known`; const positiveKey = `${auditState.target}_positive`;
  const body = byId("datasetBody"); body.replaceChildren();
  auditState.payload.datasets.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(row.dataset_key)}</td><td>${integer(row.rows)}</td><td>${integer(row.records)}</td><td>${integer(row.subjects)}</td><td>${integer(row.lineage_groups)}</td><td>${integer(row[targetKey])}</td><td>${integer(row[positiveKey])}</td>`;
    body.append(tr);
  });
}

function metricOrAccuracy(row) { return row.balanced_accuracy == null ? row.accuracy : row.balanced_accuracy; }
function percent(value) { return value == null || !Number.isFinite(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(1)}%`; }
function integer(value) { return Number(value || 0).toLocaleString("en-US"); }
function number(value) { return value == null ? "—" : Number(value).toFixed(3); }
function seconds(value) { return value == null ? "—" : `${Number(value).toFixed(1)} s`; }
function bytes(value) { const units = ["B", "KB", "MB", "GB", "TB"]; let n = Number(value || 0); let i = 0; while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; } return `${n.toFixed(i ? 2 : 0)} ${units[i]}`; }
function escapeHtml(value) { const div = document.createElement("div"); div.textContent = String(value ?? "—"); return div.innerHTML; }
