"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const sessionId = browserSessionId();

function browserSessionId() {
  const key = "ecg-cascade-browser-session-v1";
  try {
    const existing = localStorage.getItem(key);
    if (existing && /^[A-Za-z0-9_-]{1,80}$/.test(existing)) return existing;
    const created = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(key, created);
    return created;
  } catch {
    return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
}

const COLORS = {
  ink: "#29424e", muted: "#829198", grid: "#e7eeee", minorGrid: "#f0f4f4",
  teal: "#0c9c8b", tealSoft: "rgba(12,156,139,.12)", blue: "#3978a8",
  amber: "#d68732", purple: "#7559c7", purpleSoft: "rgba(117,89,199,.12)",
  quality: "#1689a2", conduction: "#b45b78",
  unsupported: "#b9c4c8", white: "#ffffff",
};

const state = {
  source: null,
  sourceToken: null,
  preview: null,
  analysis: null,
  playing: false,
  cursorS: 0,
  lastFrameMs: null,
  raf: null,
  toastTimer: null,
  matrixSignature: "",
  matrixGroup: "all",
  capabilities: {
    uploads: ["127.0.0.1", "localhost"].includes(window.location.hostname),
    max_duration_s: 300,
    deployment: ["127.0.0.1", "localhost"].includes(window.location.hostname) ? "local" : "hosted",
  },
};

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  setupCanvases();
  drawAllEmpty();
  applyCapabilities(state.capabilities);
  loadRuntimeCapabilities();
});

function bindControls() {
  $("#demoButton").addEventListener("click", loadPreparedDemo);
  $("#uploadButton").addEventListener("click", chooseUploadFiles);
  $("#fileInput").addEventListener("change", (event) => uploadFiles([...event.target.files]));
  $("#previewButton").addEventListener("click", refreshPreview);
  $("#analyzeButton").addEventListener("click", analyzeSignal);
  $("#calibrationBeatsInput").addEventListener("input", validateCalibrationInput);
  $("#playButton").addEventListener("click", togglePlayback);
  $("#previousBeatButton").addEventListener("click", () => stepBeat(-1));
  $("#nextBeatButton").addEventListener("click", () => stepBeat(1));
  $("#timeline").addEventListener("input", scrubTimeline);
  $("#speedSelect").addEventListener("change", () => { state.lastFrameMs = null; });
  $("#exportButton").addEventListener("click", exportMatrixCsv);
  $("#matrixGroupSelect").addEventListener("change", (event) => {
    state.matrixGroup = event.target.value;
    state.matrixSignature = "";
    renderAtCursor();
  });
  $("#layoutToggle").addEventListener("click", toggleLayoutMode);
  $("#glossaryButton").addEventListener("click", () => $("#glossaryDialog").showModal());
  $("#closeGlossary").addEventListener("click", () => $("#glossaryDialog").close());
  $("#glossaryDialog").addEventListener("click", (event) => {
    if (event.target === $("#glossaryDialog")) $("#glossaryDialog").close();
  });
  $("#presenterGuideButton").addEventListener("click", () => $("#presenterDialog").showModal());
  $("#closePresenterGuide").addEventListener("click", () => $("#presenterDialog").close());
  $("#presenterDoneButton").addEventListener("click", () => $("#presenterDialog").close());
  $("#presenterDialog").addEventListener("click", (event) => {
    if (event.target === $("#presenterDialog")) $("#presenterDialog").close();
  });
}

function chooseUploadFiles() {
  if (state.capabilities.uploads) {
    $("#fileInput").click();
    return;
  }
  showToast(
    "Hosted ECG uploads need file storage",
    "Neon is saving session results, but raw EDF/WFDB/CSV files need secure object storage. Use the prepared example for now; local mode still accepts private files.",
    true,
  );
}

async function loadRuntimeCapabilities() {
  try {
    const payload = await api("/api/health");
    applyCapabilities(payload.capabilities);
  } catch {
    // Keep the conservative hostname-based defaults if the health check fails.
  }
}

function toggleLayoutMode() {
  const compact = document.body.classList.toggle("one-screen");
  const button = $("#layoutToggle");
  button.setAttribute("aria-pressed", String(compact));
  button.innerHTML = compact ? "<span>↗</span> Detailed view" : "<span>⊞</span> Fit one screen";
  window.scrollTo({ top: 0, behavior: "instant" });
  requestAnimationFrame(renderAtCursor);
}

function setupCanvases() {
  const observer = new ResizeObserver(() => renderAtCursor());
  ["#signalCanvas", "#calibrationCanvas", "#morphCanvas", "#templateCanvas", "#pqrstCanvas", "#hrvCanvas", "#prsaCanvas", "#qualityCanvas", "#conductionCanvas"].forEach((id) => observer.observe($(id)));
}

async function api(path, options = {}) {
  const url = new URL(path, window.location.href);
  const route = url.pathname.replace(/^\/api\/?/, "");
  url.pathname = "/api";
  url.searchParams.set("route", route);
  const response = await fetch(`${url.pathname}${url.search}`, options);
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error(`The ECG demo server returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
  return payload;
}

async function loadPreparedDemo() {
  stopPlayback();
  showLoading("Loading prepared ECG", "Opening MIT-BIH record 100 and preparing a clean signal preview…");
  try {
    const payload = await api("/api/demo", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session: sessionId }),
    });
    applyCapabilities(payload.capabilities);
    applySource(payload.source, payload.source_token);
    applyPreview(payload.preview);
    showToast("Example ready", "MIT-BIH record 100 is loaded. Press “Run every feature branch” for the walkthrough.");
  } catch (error) { showError(error); }
  finally { hideLoading(); }
}

async function uploadFiles(files) {
  if (!files.length) return;
  if (!state.capabilities.uploads) {
    showToast("Prepared example only", "For privacy, the public demo does not accept ECG uploads. Run the application locally to analyze private files.", true);
    return;
  }
  stopPlayback();
  showLoading("Reading ECG files", `Uploading ${files.length} local file${files.length === 1 ? "" : "s"} to the private local demo server…`);
  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      $("#loadingMessage").textContent = `Reading ${file.name} (${index + 1} of ${files.length})…`;
      await api(`/api/upload?session=${encodeURIComponent(sessionId)}&filename=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    }
    const payload = await api("/api/inspect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: sessionId, files: files.map((file) => file.name), csv_sampling_rate_hz: csvRateOrNull() }),
    });
    applySource(payload.source, null);
    await refreshPreview();
    showToast("Signal loaded", `${payload.source.name} is ready for branch analysis.`);
  } catch (error) { showError(error); }
  finally { hideLoading(); $("#fileInput").value = ""; }
}

function applyCapabilities(capabilities) {
  if (!capabilities) return;
  state.capabilities = { ...state.capabilities, ...capabilities };
  const uploadButton = $("#uploadButton");
  uploadButton.disabled = false;
  uploadButton.classList.toggle("storage-required", !state.capabilities.uploads);
  uploadButton.title = state.capabilities.uploads
    ? "Upload an EDF, WFDB, or numeric CSV signal"
    : "Hosted research demo: secure object storage is required for raw ECG uploads";
  $("#uploadAvailabilityNote").textContent = state.capabilities.uploads
    ? "EDF · WFDB (.hea + .dat) · numeric CSV"
    : "Public demo: prepared example only · raw-file storage is not connected";
  const maxDuration = Number(state.capabilities.max_duration_s) || 300;
  $("#durationInput").max = String(maxDuration);
}

function applySource(source, sourceToken = null) {
  state.source = source;
  state.sourceToken = sourceToken;
  state.analysis = null;
  state.matrixSignature = "";
  $("#sourceCard").classList.remove("is-empty");
  $("#sourceKind").textContent = `${source.kind} SIGNAL`;
  $("#sourceName").textContent = source.name;
  const duration = source.duration_s == null ? "duration from sampling rate" : `${formatDuration(source.duration_s)} total`;
  const rate = source.sampling_rate_hz == null ? "sampling rate needed" : `${formatNumber(source.sampling_rate_hz, 1)} Hz`;
  $("#sourceMeta").textContent = `${source.channels.length} channel${source.channels.length === 1 ? "" : "s"} · ${rate} · ${duration}`;
  const channelSelect = $("#channelSelect");
  channelSelect.replaceChildren(...source.channels.map((channel) => new Option(channel, channel)));
  const preferred = source.channels.find((channel) => /(^|[^a-z])(ecg|ekg|mlii)([^a-z]|$)/i.test(channel));
  if (preferred) channelSelect.value = preferred;
  channelSelect.disabled = false;
  $("#startInput").disabled = false;
  $("#durationInput").disabled = false;
  const maxDuration = Number(state.capabilities.max_duration_s) || 300;
  $("#durationInput").value = Math.max(3, Math.min(120, maxDuration, Math.floor(source.duration_s || 120)));
  $("#samplingRateField").hidden = source.kind !== "CSV";
  if (source.sampling_rate_hz) $("#samplingRateInput").value = formatNumber(source.sampling_rate_hz, 4);
  $("#previewButton").disabled = false;
  validateCalibrationInput();
  resetAnalysisUi();
}

function validateCalibrationInput() {
  const input = $("#calibrationBeatsInput");
  const value = Number(input.value);
  const valid = Number.isInteger(value) && value >= 5 && value <= 100;
  input.classList.toggle("invalid", !valid);
  input.setAttribute("aria-invalid", String(!valid));
  input.setCustomValidity(valid ? "" : "Varon morphology requires 5 to 100 calibration beats.");
  $("#calibrationRule").classList.toggle("invalid", !valid);
  $("#calibrationRule").textContent = valid ? "minimum 5" : "Varon needs 5–100";
  $("#analyzeButton").disabled = !state.source || !valid;
  return valid;
}

async function refreshPreview() {
  if (!state.source) return;
  try {
    const payload = await api("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestBody()),
    });
    applyPreview(payload.preview);
  } catch (error) { showError(error); }
}

function applyPreview(preview) {
  state.preview = preview;
  $("#signalEmpty").hidden = true;
  $("#signalWindowLabel").textContent = `· ${preview.channel} · ${formatNumber(preview.sampling_rate_hz, 1)} Hz preview`;
  drawPreviewSignal();
}

async function analyzeSignal() {
  if (!state.source) return;
  if (!validateCalibrationInput()) {
    showToast("Five calibration beats required", "Varon stacks five aligned QRS complexes, so choose a whole number from 5 to 100.", true);
    return;
  }
  stopPlayback();
  showLoading("Running every extraction branch", "Detecting R peaks, measuring morphology and RR/HRV, auditing signal quality, calculating conduction timing, and causally joining the matrix…");
  try {
    const payload = await api("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...requestBody(), timestamp_track: selectedTrack(), calibration_beats: Number($("#calibrationBeatsInput").value) }),
    });
    state.analysis = payload.analysis;
    state.cursorS = payload.analysis.source.start_s;
    state.matrixSignature = "";
    initializeAnalysisUi();
    populateGlossary();
    renderAtCursor();
    startPlayback();
    showToast("Extraction complete", `${payload.analysis.summary.r_peak_count} heartbeat anchors were processed. The dashboard is replaying the extraction now.`);
  } catch (error) { showError(error); }
  finally { hideLoading(); }
}

function requestBody() {
  return {
    session: sessionId,
    source_token: state.sourceToken,
    channel: $("#channelSelect").value,
    start_s: Number($("#startInput").value),
    duration_s: Number($("#durationInput").value),
    csv_sampling_rate_hz: csvRateOrNull(),
  };
}

function csvRateOrNull() {
  if (!state.source || state.source.kind !== "CSV") return null;
  const value = Number($("#samplingRateInput").value);
  return Number.isFinite(value) ? value : null;
}

function selectedTrack() { return $("input[name='track']:checked").value; }

function initializeAnalysisUi() {
  const analysis = state.analysis;
  const calibration = analysis.morphology_detail.patient_calibration;
  $("#signalEmpty").hidden = true;
  $("#signalWindowLabel").textContent = `· ${analysis.source.channel} · live 12-second view`;
  $("#playButton").disabled = false;
  $("#previousBeatButton").disabled = false;
  $("#nextBeatButton").disabled = false;
  $("#timeline").disabled = false;
  $("#speedSelect").disabled = false;
  $("#exportButton").disabled = false;
  $("#durationLabel").textContent = formatClock(analysis.source.duration_s);
  setStatus($("#morphStatus"), "running", "Building");
  setStatus($("#hrvStatus"), "running", "Building");
  setStatus($("#qualityStatus"), "running", "Opening 10 s window");
  setStatus($("#conductionStatus"), "running", "Waiting for landmarks");
  $("#qualityUnitNote").textContent = analysis.signal_quality.amplitude_calibrated
    ? `${analysis.signal_quality.input_unit} input is calibrated to physical amplitude`
    : `Input unit is unknown; amplitude-valued SQIs stay unavailable`;
  $("#calibrationProvenance").textContent = `${calibration.boundary_source} · ${calibration.lead_name} · patient/record specific`;
  $("#calProgressMetric").textContent = `0 / ${calibration.requested_beats}`;
  $("#templateBuildMetric").textContent = `0 / ${analysis.morphology_detail.patient_template.requested_calibration_beats}`;
  $("#calFreezeMetric").textContent = calibration.frozen_at_time_s == null ? "unavailable" : formatClock(calibration.frozen_at_time_s - analysis.source.start_s);
  $("#stageSignal").className = "workflow-step complete";
  $("#stagePeaks").className = "workflow-step active";
}

function resetAnalysisUi() {
  stopPlayback();
  state.analysis = null;
  $("#playButton").disabled = true;
  $("#previousBeatButton").disabled = true;
  $("#nextBeatButton").disabled = true;
  $("#timeline").disabled = true;
  $("#timeline").value = 0;
  $("#timeline").style.setProperty("--range-progress", "0%");
  $("#speedSelect").disabled = true;
  $("#exportButton").disabled = true;
  $("#elapsedLabel").textContent = "0:00";
  $("#durationLabel").textContent = "0:00";
  $("#cursorLabel").textContent = "Ready to analyze";
  $("#morphBeatMetric").textContent = "0 / 5";
  $("#morphWindowMetric").textContent = "0";
  $("#lambdaMetric").textContent = "—";
  $("#heartRateMetric").textContent = "—";
  $("#hrvBuildMetric").textContent = "0 / 100";
  $("#sd1Metric").textContent = "—";
  $("#sd2Metric").textContent = "—";
  $("#csiMetric").textContent = "—";
  $("#jointMetric").textContent = "—";
  $("#prsaBuildMetric").textContent = "0 / 80";
  $("#prsaRrMetric").textContent = "—";
  $("#prsaSlopeMetric").textContent = "—";
  $("#prsaDeltaMetric").textContent = "—";
  $("#bprsaSlopeMetric").textContent = "—";
  $("#bprsaDeltaMetric").textContent = "—";
  $("#prsaAnchorMetric").textContent = "—";
  $("#prsaSupportMetric").textContent = "—";
  $("#calProgressMetric").textContent = `0 / ${$("#calibrationBeatsInput").value || 20}`;
  $("#calPreMetric").textContent = "—";
  $("#calPostMetric").textContent = "—";
  $("#calWindowMetric").textContent = "—";
  $("#calFreezeMetric").textContent = "—";
  $("#inlineCropMetric").textContent = "—";
  $("#templateBuildMetric").textContent = `0 / ${$("#calibrationBeatsInput").value || 20}`;
  $("#templateCountMetric").textContent = "—";
  $("#templateCorrMetric").textContent = "—";
  $("#templateDtwMetric").textContent = "—";
  $("#qrsDurationMetric").textContent = "—";
  $("#prMetric").textContent = "—";
  $("#qtMetric").textContent = "—";
  $("#polarityMetric").textContent = "—";
  ["#qualityFiniteMetric", "#qualityFlatMetric", "#qualitySpectrumMetric", "#qualityAgreementMetric", "#qualityRrMetric", "#qualityTemplateMetric", "#qualityAvailabilityMetric", "#conductionPrMetric", "#conductionQrsMetric", "#conductionQtMetric", "#conductionTpeMetric", "#conductionBazettMetric", "#conductionFridericiaMetric", "#conductionFraminghamMetric", "#conductionDeltaMetric"].forEach((id) => $(id).textContent = "—");
  $("#qualityWindowMetric").textContent = "0 / 10 s";
  $("#qualityReasonList").innerHTML = '<span class="availability-chip pending">Run the branch to see prerequisites</span>';
  $("#qualityUnitNote").textContent = "Waiting for signal-unit metadata";
  $("#conductionFiveMetric").textContent = "Building 0 / 5";
  $("#conductionNineMetric").textContent = "Building 0 / 9";
  $("#conductionThirtyMetric").textContent = "Building 0 / 30";
  $("#conductionFiveCopy").textContent = "Median, range and robust slope appear after enough automatic QT measurements.";
  $("#conductionThirtyCopy").textContent = "SD/RMSSD and exploratory context require a complete trailing history.";
  $("#calibrationState").className = "freeze-state";
  $("#calibrationState").textContent = "Waiting for analysis";
  ["#calStepCollect", "#calStepSummarize", "#calStepFreeze", "#calStepEvaluate"].forEach((id) => $(id).classList.remove("active", "complete"));
  $("#matrixRowMetric").textContent = "0";
  setStatus($("#morphStatus"), "pending", "Waiting");
  setStatus($("#hrvStatus"), "pending", "Waiting");
  setStatus($("#qualityStatus"), "pending", "Waiting");
  setStatus($("#conductionStatus"), "pending", "Waiting");
  ["#stageSignal", "#stagePeaks", "#stageMorph", "#stageHrv", "#stageQuality", "#stageConduction", "#stageMatrix"].forEach((id, index) => {
    $(id).classList.remove("active", "complete");
    if (index === 0) $(id).classList.add("active");
  });
  renderMatrix([]);
  drawMorphology(null);
  drawCalibration(0);
  drawTemplate(null);
  drawPqrst(null);
  drawHrv([]);
  drawPrsaBprsa(null);
  drawQuality([]);
  drawConduction([]);
  updateGuide({ eventCount: 0, calibrationCount: 0, qualityCount: 0, conductionCount: 0, fusionRows: [] });
}

function startPlayback() {
  if (!state.analysis) return;
  const end = analysisEndS();
  if (state.cursorS >= end - 1e-6) state.cursorS = state.analysis.source.start_s;
  state.playing = true;
  state.lastFrameMs = null;
  $("#playIcon").textContent = "Ⅱ";
  state.raf = requestAnimationFrame(playbackTick);
}

function stopPlayback() {
  state.playing = false;
  state.lastFrameMs = null;
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = null;
  if ($("#playIcon")) $("#playIcon").textContent = "▶";
}

function togglePlayback() { state.playing ? stopPlayback() : startPlayback(); }

function playbackTick(frameMs) {
  if (!state.playing || !state.analysis) return;
  if (state.lastFrameMs != null) {
    const deltaS = Math.min(.15, (frameMs - state.lastFrameMs) / 1000);
    state.cursorS += deltaS * Number($("#speedSelect").value);
  }
  state.lastFrameMs = frameMs;
  const end = analysisEndS();
  if (state.cursorS >= end) {
    state.cursorS = end;
    renderAtCursor();
    stopPlayback();
    $("#cursorLabel").textContent = "Playback complete · drag the timeline to review";
    return;
  }
  renderAtCursor();
  state.raf = requestAnimationFrame(playbackTick);
}

function scrubTimeline(event) {
  if (!state.analysis) return;
  const progress = Number(event.target.value) / 1000;
  state.cursorS = state.analysis.source.start_s + progress * state.analysis.source.duration_s;
  state.lastFrameMs = null;
  renderAtCursor();
}

function stepBeat(direction) {
  if (!state.analysis?.events?.length) return;
  stopPlayback();
  const times = state.analysis.events.map((event) => event.time_s);
  const tolerance = 1e-4;
  let index;
  if (direction > 0) index = upperBoundValues(times, state.cursorS + tolerance);
  else index = lowerBound(times, state.cursorS - tolerance) - 1;
  index = clamp(index, 0, times.length - 1);
  state.cursorS = times[index];
  renderAtCursor();
  $("#cursorLabel").textContent = `Beat ${index + 1} of ${times.length} · paused for inspection`;
}

function renderAtCursor() {
  if (!state.analysis) {
    if (state.preview) drawPreviewSignal();
    return;
  }
  const analysis = state.analysis;
  const cursor = state.cursorS;
  drawLiveSignal(analysis, cursor);
  const eventCount = upperBound(analysis.events, cursor, "time_s");
  const rrCount = upperBound(analysis.rr, cursor, "time_s");
  const hrvCount = upperBound(analysis.hrv, cursor, "time_s");
  const prsaRows = analysis.prsa_bprsa?.rows || [];
  const prsaCount = upperBound(prsaRows, cursor, "time_s");
  const fusionCount = upperBound(analysis.fusion, cursor, "time_s");
  const calibration = analysis.morphology_detail.patient_calibration;
  const calibrationCount = upperBound(calibration.beats, cursor, "offset_time_s");
  const calibrationFrozen = calibration.frozen_at_time_s != null && cursor >= calibration.frozen_at_time_s;
  const rawMorphCount = upperBound(analysis.morphology, cursor, "time_s");
  const morphCount = calibrationFrozen ? rawMorphCount : 0;
  const templateRows = analysis.morphology_detail.patient_template.features || [];
  const templateCount = upperBound(templateRows, cursor, "time_s");
  const pqrstRows = analysis.morphology_detail.pqrst.rows || [];
  const pqrstCount = upperBound(pqrstRows, cursor, "time_s");
  const qualityRows = analysis.signal_quality?.rows || [];
  const qualityCount = upperBound(qualityRows, cursor, "time_s");
  const conductionBeats = analysis.conduction?.beats || [];
  const conductionCount = upperBound(conductionBeats, cursor, "r_peak_time_s");
  const conductionFiveRows = (analysis.conduction?.five_beat || []).filter((row) => row.measurement === "qt_interval" && row.r_peak_time_s <= cursor);
  const conductionNineRows = (analysis.conduction?.nine_beat || []).filter((row) => row.r_peak_time_s <= cursor);
  const conductionThirtyRows = (analysis.conduction?.thirty_beat || []).filter((row) => row.r_peak_time_s <= cursor);
  const latestMorph = morphCount ? analysis.morphology[morphCount - 1] : null;
  const latestTemplate = templateCount ? templateRows[templateCount - 1] : null;
  const latestPqrst = pqrstCount ? pqrstRows[pqrstCount - 1] : null;
  const latestHrv = hrvCount ? analysis.hrv[hrvCount - 1] : null;
  const latestPrsa = prsaCount ? prsaRows[prsaCount - 1] : null;
  const latestQuality = qualityCount ? qualityRows[qualityCount - 1] : null;
  const latestConduction = conductionCount ? conductionBeats[conductionCount - 1] : null;
  const latestConductionFive = conductionFiveRows.at(-1) || null;
  const latestConductionNine = conductionNineRows.at(-1) || null;
  const latestConductionThirty = conductionThirtyRows.at(-1) || null;
  const visibleRr = analysis.rr.slice(0, rrCount);
  drawCalibration(calibrationCount);
  drawMorphology(latestMorph);
  drawTemplate(latestTemplate);
  drawPqrst(latestPqrst);
  drawHrv(visibleRr);
  drawPrsaBprsa(latestPrsa);
  drawQuality(qualityRows.slice(0, qualityCount));
  drawConduction(conductionBeats.slice(0, conductionCount));
  renderMatrix(analysis.fusion.slice(0, fusionCount));
  updateMetrics({ eventCount, rrCount, morphCount, latestMorph, latestTemplate, latestPqrst, latestHrv, latestPrsa, latestQuality, qualityCount, latestConduction, latestConductionFive, latestConductionNine, latestConductionThirty, conductionCount, calibrationCount, templateCount, fusionRows: analysis.fusion.slice(0, fusionCount) });
  updateTimeline();
  updateWorkflow({ eventCount, morphCount, latestHrv, latestPrsa, qualityCount, conductionCount, fusionRows: analysis.fusion.slice(0, fusionCount) });
  updateGuide({ eventCount, calibrationCount, qualityCount, conductionCount, fusionRows: analysis.fusion.slice(0, fusionCount) });
}

function updateMetrics({ eventCount, rrCount, morphCount, latestMorph, latestTemplate, latestPqrst, latestHrv, latestPrsa, latestQuality, qualityCount, latestConduction, latestConductionFive, latestConductionNine, latestConductionThirty, conductionCount, calibrationCount, templateCount, fusionRows }) {
  const calibration = state.analysis.morphology_detail.patient_calibration;
  const template = state.analysis.morphology_detail.patient_template;
  $("#morphBeatMetric").textContent = `${morphCount ? Math.min(5, eventCount) : 0} / 5`;
  $("#morphWindowMetric").textContent = String(morphCount);
  $("#lambdaMetric").textContent = latestMorph?.lambda1 == null ? "—" : compactNumber(latestMorph.lambda1);
  const calibrationFrozen = calibration.frozen_at_time_s != null && state.cursorS >= calibration.frozen_at_time_s;
  $("#inlineCropMetric").textContent = calibrationFrozen && calibration.applied_pre_r_ms != null ? `−${formatNumber(calibration.applied_pre_r_ms, 1)} / +${formatNumber(calibration.applied_post_r_ms, 1)}` : "—";
  const visibleCalibration = calibration.beats.slice(0, calibrationCount);
  const runningPre = visibleCalibration.map((beat) => beat.onset_to_r_ms).sort((a,b)=>a-b);
  const runningPost = visibleCalibration.map((beat) => beat.r_to_offset_ms).sort((a,b)=>a-b);
  $("#calProgressMetric").textContent = `${calibrationCount} / ${calibration.requested_beats}`;
  $("#calPreMetric").textContent = runningPre.length ? `${formatNumber(quantile(runningPre, .95), 1)} ms` : "—";
  $("#calPostMetric").textContent = runningPost.length ? `${formatNumber(quantile(runningPost, .95), 1)} ms` : "—";
  $("#calWindowMetric").textContent = calibrationFrozen && calibration.applied_pre_r_ms != null ? `−${formatNumber(calibration.applied_pre_r_ms, 1)} / +${formatNumber(calibration.applied_post_r_ms, 1)} ms` : "not frozen";
  updateCalibrationState(calibrationCount, calibrationFrozen);
  const templateCalibrationVisible = templateRowsBeforeCursor(template).filter((row) => row.calibration_member).length;
  $("#templateBuildMetric").textContent = `${Math.min(template.used_calibration_beats || 0, templateCalibrationVisible)} / ${template.requested_calibration_beats}`;
  $("#templateCountMetric").textContent = template.available && templateCalibrationVisible >= template.used_calibration_beats ? String(template.template_count) : "—";
  $("#templateCorrMetric").textContent = latestTemplate?.evaluation_eligible && latestTemplate.template_correlation != null ? formatNumber(latestTemplate.template_correlation, 3) : "—";
  $("#templateDtwMetric").textContent = latestTemplate?.evaluation_eligible && latestTemplate.template_derivative_dtw_cost != null ? formatNumber(latestTemplate.template_derivative_dtw_cost, 4) : "—";
  $("#qrsDurationMetric").textContent = latestPqrst?.qrs_duration_ms == null ? "—" : `${formatNumber(latestPqrst.qrs_duration_ms, 1)} ms`;
  $("#prMetric").textContent = latestPqrst?.pr_interval_ms == null ? "—" : `${formatNumber(latestPqrst.pr_interval_ms, 1)} ms`;
  $("#qtMetric").textContent = latestPqrst?.qt_interval_ms == null ? "—" : `${formatNumber(latestPqrst.qt_interval_ms, 1)} ms`;
  $("#polarityMetric").textContent = latestPqrst?.qrs_polarity || "—";
  const latestRr = rrCount ? state.analysis.rr[rrCount - 1] : null;
  $("#heartRateMetric").textContent = latestRr ? `${formatNumber(latestRr.heart_rate_bpm, 0)} bpm` : "—";
  $("#hrvBuildMetric").textContent = `${Math.min(100, latestHrv?.window_rr_count || rrCount)} / 100`;
  $("#sd1Metric").textContent = latestHrv?.sd1_ms == null ? "—" : `${formatNumber(latestHrv.sd1_ms, 1)} ms`;
  $("#sd2Metric").textContent = latestHrv?.sd2_ms == null ? "—" : `${formatNumber(latestHrv.sd2_ms, 1)} ms`;
  $("#csiMetric").textContent = latestHrv?.csi == null ? "—" : `${formatNumber(latestHrv.csi, 2)} / ${formatNumber(latestHrv.modcsi_ms, 1)}`;
  $("#jointMetric").textContent = latestHrv?.j1 == null ? "—" : `${formatNumber(latestHrv.j1, 2)} / ${formatNumber(latestHrv.j2, 2)}`;
  $("#prsaBuildMetric").textContent = `${Math.min(80, latestPrsa?.window_beat_count || rrCount)} / 80`;
  $("#prsaRrMetric").textContent = latestPrsa?.mean_rr80_ms == null ? "—" : `${formatNumber(latestPrsa.mean_rr80_ms, 1)} / ${formatNumber(latestPrsa.sdnn80_ms, 1)} ms`;
  $("#prsaSlopeMetric").textContent = latestPrsa?.prsa_s_rr_ms_per_sample == null ? "—" : formatNumber(latestPrsa.prsa_s_rr_ms_per_sample, 3);
  $("#prsaDeltaMetric").textContent = latestPrsa?.prsa_delta_rr_ms_per_sample == null ? "—" : formatNumber(latestPrsa.prsa_delta_rr_ms_per_sample, 3);
  $("#bprsaSlopeMetric").textContent = latestPrsa?.bprsa_s_r_ms_per_sample == null ? "—" : formatNumber(latestPrsa.bprsa_s_r_ms_per_sample, 3);
  $("#bprsaDeltaMetric").textContent = latestPrsa?.bprsa_delta_r_ms_per_sample == null ? "—" : formatNumber(latestPrsa.bprsa_delta_r_ms_per_sample, 3);
  $("#prsaAnchorMetric").textContent = latestPrsa?.defined ? `${latestPrsa.prsa_anchor_count} / ${latestPrsa.bprsa_anchor_count}` : "—";
  $("#prsaSupportMetric").textContent = latestPrsa?.rr_coverage == null ? "—" : (latestPrsa.numerical_quality_pass ? `${formatNumber(100 * latestPrsa.rr_coverage, 0)}% RR / ${formatNumber(100 * latestPrsa.r_amplitude_coverage, 0)}% R-amp` : `failed · ${latestPrsa.resampled_rr_nonpositive_count} nonpositive spline RR`);
  const elapsed = Math.max(0, state.cursorS - state.analysis.source.start_s);
  $("#qualityWindowMetric").textContent = latestQuality ? `10 / 10 s · window ${qualityCount}` : `${formatNumber(Math.min(10, elapsed), 1)} / 10 s`;
  $("#qualityFiniteMetric").textContent = qualityValue(latestQuality, "finite_fraction", true);
  $("#qualityFlatMetric").textContent = qualityValue(latestQuality, "flat_fraction", true);
  $("#qualitySpectrumMetric").textContent = qualityPair(latestQuality, "bassqi_clifford", "psqi_clifford");
  $("#qualityAgreementMetric").textContent = qualityPair(latestQuality, "bsqi_li2008_jaccard_150ms", "qsqi_zhao2018_dice_75ms");
  $("#qualityRrMetric").textContent = qualityValue(latestQuality, "rr_support_ho2024_150ms", true);
  $("#qualityTemplateMetric").textContent = qualityValue(latestQuality, "template_corr_orphanidou");
  $("#qualityAvailabilityMetric").textContent = latestQuality ? `${latestQuality.available_count} / ${latestQuality.feature_count}` : "—";
  updateQualityAvailability(latestQuality);

  $("#conductionPrMetric").textContent = milliseconds(latestConduction?.pr_interval_ms);
  $("#conductionQrsMetric").textContent = milliseconds(latestConduction?.qrs_duration_ms);
  $("#conductionQtMetric").textContent = latestConduction?.qt_interval_ms == null ? "—" : `${formatNumber(latestConduction.qt_interval_ms, 1)} / ${formatNumber(latestConduction.jt_interval_ms, 1)} ms`;
  $("#conductionTpeMetric").textContent = milliseconds(latestConduction?.t_peak_to_end_ms);
  $("#conductionBazettMetric").textContent = milliseconds(latestConduction?.qtc_bazett_ms);
  $("#conductionFridericiaMetric").textContent = milliseconds(latestConduction?.qtc_fridericia_ms);
  $("#conductionFraminghamMetric").textContent = milliseconds(latestConduction?.qtc_framingham_ms);
  $("#conductionDeltaMetric").textContent = latestConduction?.delta_pr_interval_ms == null ? "—" : `${signedNumber(latestConduction.delta_pr_interval_ms)} / ${signedNumber(latestConduction.delta_qt_interval_ms)} ms`;
  const fiveHistory = latestConductionFive?.history_beats5 || Math.min(5, conductionCount);
  $("#conductionFiveMetric").textContent = latestConductionFive?.summary_defined ? `QT median ${formatNumber(latestConductionFive.median_ms, 1)} ms · range ${formatNumber(latestConductionFive.range_ms, 1)} ms` : `Building ${fiveHistory} / 5`;
  $("#conductionFiveCopy").textContent = latestConductionFive?.summary_defined ? `Robust QT slope: ${signedNumber(latestConductionFive.theil_sen_slope_ms_per_beat)} ms/beat. Descriptive only; no cause is assigned.` : "Median, range and robust slope appear after enough automatic QT measurements.";
  $("#conductionNineMetric").textContent = latestConductionNine?.qt_rr_mean9_defined ? `QTc Fridericia ${formatNumber(latestConductionNine.qtc_fridericia_from_means9_ms, 1)} ms` : `Building ${Math.min(9, conductionCount)} / 9`;
  $("#conductionThirtyMetric").textContent = latestConductionThirty?.qt_interval_variability30_defined ? `QT SD ${formatNumber(latestConductionThirty.qt_interval_sd30_ms, 1)} · RMSSD ${formatNumber(latestConductionThirty.qt_interval_rmssd30_ms, 1)} ms` : `Building ${latestConductionThirty?.history_beats30 || Math.min(30, conductionCount)} / 30`;
  $("#conductionThirtyCopy").textContent = latestConductionThirty?.qt_interval_variability30_defined ? "A complete causal history exists. Variability remains exploratory because automatic landmark error can be similar to true beat-to-beat change." : "SD/RMSSD and exploratory context require a complete trailing history.";
  const complete = fusionRows.filter((row) => row.measurements_defined).length;
  $("#matrixRowMetric").textContent = String(complete);
  const replayComplete = state.cursorS >= analysisEndS() - 1e-6;
  if (morphCount) setStatus($("#morphStatus"), "ready", replayComplete ? "Complete" : "Extracting");
  if (latestHrv?.defined && latestPrsa?.defined) setStatus($("#hrvStatus"), "ready", "Both lanes ready");
  else if (latestPrsa?.defined) setStatus($("#hrvStatus"), "ready", "80-beat PRSA ready");
  if (latestQuality) setStatus($("#qualityStatus"), "ready", replayComplete ? "Complete" : `${latestQuality.available_count} computable`);
  if (latestConduction) setStatus($("#conductionStatus"), "ready", replayComplete ? "Complete" : "Information visible");
}

function qualityValue(row, key, percent = false) {
  if (!row?.available?.[key] || row.values[key] == null) return "—";
  return percent ? `${formatNumber(100 * row.values[key], 1)}%` : formatNumber(row.values[key], 3);
}

function qualityPair(row, left, right) {
  const a = row?.available?.[left] ? formatNumber(row.values[left], 3) : "—";
  const b = row?.available?.[right] ? formatNumber(row.values[right], 3) : "—";
  return `${a} / ${b}`;
}

function updateQualityAvailability(row) {
  const list = $("#qualityReasonList");
  if (!row) { list.innerHTML = '<span class="availability-chip pending">A complete 10-second window is required</span>'; return; }
  const preferred = ["finite_fraction", "flat_fraction", "bassqi_clifford", "psqi_clifford", "bsqi_li2008_jaccard_150ms", "qsqi_zhao2018_dice_75ms", "rr_support_ho2024_150ms", "template_corr_orphanidou", "rail_fraction", "isqi_li2008_max_jaccard_150ms", "mains_rms_galeotti", "hf_rms_uv"];
  list.replaceChildren(...preferred.map((key) => {
    const chip = document.createElement("span");
    chip.className = `availability-chip${row.available[key] ? "" : " unavailable"}`;
    chip.textContent = row.available[key] ? `${qualityLabel(key)} ✓` : `${qualityLabel(key)} · ${humanReason(row.reasons[key])}`;
    return chip;
  }));
}

function qualityLabel(key) { return ({finite_fraction:"finite",flat_fraction:"flat",bassqi_clifford:"basSQI",psqi_clifford:"pSQI",bsqi_li2008_jaccard_150ms:"bSQI",qsqi_zhao2018_dice_75ms:"qSQI",rr_support_ho2024_150ms:"RR support",template_corr_orphanidou:"beat correlation",rail_fraction:"ADC rail",isqi_li2008_max_jaccard_150ms:"other-lead iSQI",mains_rms_galeotti:"mains RMS",hf_rms_uv:"HF RMS"})[key] || key; }
function humanReason(reason) { return String(reason || "unavailable").replaceAll("_", " "); }
function milliseconds(value) { return value == null ? "—" : `${formatNumber(value, 1)} ms`; }
function signedNumber(value) { return value == null ? "—" : `${value > 0 ? "+" : ""}${formatNumber(value, 1)}`; }

function templateRowsBeforeCursor(template) {
  if (!template?.features?.length) return [];
  return template.features.slice(0, upperBound(template.features, state.cursorS, "time_s"));
}

function updateCalibrationState(count, frozen) {
  const calibration = state.analysis.morphology_detail.patient_calibration;
  const status = $("#calibrationState");
  const steps = [$("#calStepCollect"), $("#calStepSummarize"), $("#calStepFreeze"), $("#calStepEvaluate")];
  steps.forEach((step) => step.classList.remove("active", "complete"));
  if (!count) {
    status.className = "freeze-state collecting"; status.textContent = "Waiting for first eligible QRS"; steps[0].classList.add("active"); return;
  }
  if (!frozen) {
    if (state.cursorS >= analysisEndS() - 1e-6 && count < calibration.requested_beats) {
      status.className = "freeze-state"; status.textContent = `Unavailable · only ${count} of ${calibration.requested_beats} calibration beats`;
      steps[0].classList.add("complete"); return;
    }
    status.className = "freeze-state collecting"; status.textContent = `Collecting ${count} of ${calibration.requested_beats}`;
    steps[0].classList.add("active"); if (count >= 2) steps[1].classList.add("active"); return;
  }
  steps.slice(0,3).forEach((step) => step.classList.add("complete")); steps[3].classList.add("active");
  status.className = "freeze-state frozen"; status.textContent = "Frozen patient crop · evaluating later beats";
}

function updateTimeline() {
  const analysis = state.analysis;
  const elapsed = Math.max(0, state.cursorS - analysis.source.start_s);
  const progress = clamp(elapsed / analysis.source.duration_s, 0, 1);
  $("#timeline").value = Math.round(progress * 1000);
  $("#timeline").style.setProperty("--range-progress", `${progress * 100}%`);
  $("#elapsedLabel").textContent = formatClock(elapsed);
  $("#cursorLabel").textContent = `${analysis.track.label} anchors · ${progress < 1 ? "extracting" : "complete"}`;
}

function updateWorkflow({ eventCount, morphCount, latestHrv, latestPrsa, qualityCount, conductionCount, fusionRows }) {
  setStage("#stageSignal", true, true);
  setStage("#stagePeaks", eventCount > 0, eventCount > 4);
  setStage("#stageMorph", morphCount > 0, morphCount > 0);
  setStage("#stageHrv", (latestHrv?.window_rr_count || 0) > 0, Boolean(latestHrv?.defined && latestPrsa?.defined));
  setStage("#stageQuality", qualityCount > 0 || state.cursorS > state.analysis.source.start_s, qualityCount > 0);
  setStage("#stageConduction", conductionCount > 0, conductionCount >= 5);
  const combined = fusionRows.some((row) => row.measurements_defined);
  setStage("#stageMatrix", fusionRows.length > 0, combined);
}

function updateGuide({ eventCount, calibrationCount, qualityCount, conductionCount, fusionRows }) {
  if (!$("#guideStep")) return;
  const analysis = state.analysis;
  let step = 0, phase = "READY FOR THE WALKTHROUGH", title = "Start with the ECG itself", copy = "The trace is the only input. Run every branch, then use previous/next beat to inspect the extraction without rushing.", why = "It keeps every later number visibly connected to the waveform that produced it.";
  if (analysis) {
    const beat = Math.max(1, eventCount);
    const calibration = analysis.morphology_detail.patient_calibration;
    const frozen = calibration.frozen_at_time_s != null && state.cursorS >= calibration.frozen_at_time_s;
    if (!eventCount) {
      step = 1; phase = "STEP 1 · FOLLOW THE CURSOR"; title = "The algorithm has not reached an R peak yet"; copy = "Only raw ECG samples to the left of the amber cursor are considered visible in this replay."; why = "A causal display makes it clear which information exists at each moment.";
    } else if (!frozen) {
      step = 2; phase = `STEP 2 · BEAT ${beat}`; title = `One R peak becomes the shared anchor for every branch`; copy = `The morphology lane is collecting automatic QRS boundaries (${calibrationCount} of ${calibration.requested_beats}). RR timing starts after two anchors, conduction subtracts landmark times, and quality waits for a full 10-second window.`; why = "Sharing the timestamp aligns measurements without averaging or changing either detector’s peaks.";
    } else if (qualityCount === 0) {
      step = 3; phase = "STEP 3 · PATIENT CROP FROZEN"; title = "Calibration is now fixed before later beats are scored"; copy = "The patient-specific onset→R and R→offset extents no longer change. Later QRS groups feed λ1–λ5 while whole cycles are compared with the frozen template."; why = "Freezing separates calibration from evaluation and avoids adapting the reference to the beat being measured.";
    } else if (conductionCount < 30) {
      step = 4; phase = `STEP 4 · BEAT ${beat} · FOUR LANES ACTIVE`; title = "Watch the branches accumulate at different speeds"; copy = `Shape scores update beat by beat; RR/HRV builds toward 80- and 100-beat windows; quality window ${qualityCount} summarizes the prior 10 seconds; conduction now has ${conductionCount} beat rows and is building 5-, 9- and 30-beat context.`; why = "Different window lengths answer different questions, so missing early values are expected—not failures.";
    } else if (!fusionRows.some((row) => row.measurements_defined)) {
      step = 5; phase = "STEP 5 · CONTEXT IS READY"; title = "Quality and conduction context are visible before full HRV"; copy = "The matrix can already show contextual columns, but the core measurement gate is still waiting for all morphology and 100-RR HRV requirements."; why = "Context is preserved for inspection without silently changing which core rows are complete.";
    } else {
      step = 6; phase = "STEP 6 · CAUSAL JOIN"; title = "A complete row now combines the latest available measurements"; copy = "Morphology and RR/HRV determine core completeness. PRSA/BPRSA and signal quality remain context; conduction remains information-only. None of these values is converted into a seizure, artifact or abnormality label."; why = "The matrix is an auditable alignment of measurements, not a trained clinical prediction.";
    }
  }
  $("#guideStep").textContent = `Step ${step}`; $("#guidePhase").textContent = phase; $("#guideTitle").textContent = title; $("#guideCopy").textContent = copy; $("#guideWhy").textContent = why;
  [...$("#guideProgress").children].forEach((dot, index) => { dot.classList.toggle("complete", index < step); dot.classList.toggle("active", index === Math.min(step, 6)); });
}

function setStage(selector, active, complete) {
  const element = $(selector);
  element.classList.toggle("active", active && !complete);
  element.classList.toggle("complete", complete);
}

function drawPreviewSignal() {
  if (!state.preview) return;
  const { ctx, width, height } = canvasContext($("#signalCanvas"));
  clearCanvas(ctx, width, height);
  drawEcgGrid(ctx, width, height);
  const times = state.preview.times_s, values = state.preview.values;
  if (!times.length) return;
  const pad = { left: 52, right: 17, top: 13, bottom: 28 };
  const range = paddedRange(values, .12);
  drawPolyline(ctx, times, values, { xMin: times[0], xMax: times[times.length - 1], yMin: range[0], yMax: range[1], pad, width, height, color: COLORS.teal, lineWidth: 1.35 });
  drawAxes(ctx, width, height, pad, { xMin: times[0], xMax: times[times.length - 1], yMin: range[0], yMax: range[1], xUnit: "s", yLabel: "amplitude" });
}

function drawLiveSignal(analysis, cursor) {
  const { ctx, width, height } = canvasContext($("#signalCanvas"));
  clearCanvas(ctx, width, height); drawEcgGrid(ctx, width, height);
  const segmentStart = analysis.source.start_s, segmentEnd = analysisEndS();
  let xMin = Math.max(segmentStart, cursor - 9.5);
  let xMax = Math.min(segmentEnd, xMin + 12);
  xMin = Math.max(segmentStart, xMax - 12);
  const first = lowerBound(analysis.signal.times_s, xMin), last = upperBoundValues(analysis.signal.times_s, xMax);
  const times = analysis.signal.times_s.slice(first, last), values = analysis.signal.values.slice(first, last);
  const pad = { left: 52, right: 17, top: 13, bottom: 28 };
  const range = paddedRange(values.length ? values : [0, 1], .13);
  const plotH = height - pad.top - pad.bottom;
  const visibleEvents = analysis.events.filter((event) => event.time_s >= xMin && event.time_s <= cursor && event.time_s <= xMax);
  const maskAnchor = visibleEvents.at(-1);
  if (maskAnchor) {
    const calibration = analysis.morphology_detail.patient_calibration;
    const frozen = calibration.frozen_at_time_s != null && cursor >= calibration.frozen_at_time_s;
    if (frozen && calibration.applied_pre_r_ms != null) {
      drawMaskBand(ctx, maskAnchor.time_s - calibration.applied_pre_r_ms / 1000, maskAnchor.time_s + calibration.applied_post_r_ms / 1000, xMin, xMax, pad, width, height, "rgba(57,120,168,.10)", "#3978a8", "patient crop", 0);
    }
    const pqrstRows = analysis.morphology_detail.pqrst.rows || [];
    const qrsIndex = upperBound(pqrstRows, maskAnchor.time_s + .001, "time_s") - 1;
    const qrs = qrsIndex >= 0 ? pqrstRows[qrsIndex] : null;
    if (qrs && Math.abs(qrs.time_s - maskAnchor.time_s) < .08 && qrs.samples.qrs_onset != null && qrs.samples.qrs_offset != null) {
      const start = analysis.source.start_s + qrs.samples.qrs_onset / analysis.source.sampling_rate_hz;
      const end = analysis.source.start_s + qrs.samples.qrs_offset / analysis.source.sampling_rate_hz;
      drawMaskBand(ctx, start, end, xMin, xMax, pad, width, height, "rgba(12,156,139,.12)", "#0c9c8b", "auto QRS", 10);
    }
  }
  drawPolyline(ctx, times, values, { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color: COLORS.teal, lineWidth: 1.4 });
  for (const event of analysis.events) {
    if (event.time_s < xMin || event.time_s > cursor || event.time_s > xMax) continue;
    const x = map(event.time_s, xMin, xMax, pad.left, width - pad.right);
    ctx.strokeStyle = event.supported ? COLORS.blue : COLORS.unsupported; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.top + 2); ctx.lineTo(x, pad.top + 9); ctx.stroke();
    ctx.fillStyle = event.supported ? COLORS.blue : COLORS.unsupported;
    ctx.beginPath(); ctx.arc(x, pad.top + 3, 2.6, 0, Math.PI * 2); ctx.fill();
  }
  if (cursor >= xMin && cursor <= xMax) {
    const x = map(cursor, xMin, xMax, pad.left, width - pad.right);
    ctx.fillStyle = "rgba(214,135,50,.08)"; ctx.fillRect(x, pad.top, Math.max(0, width - pad.right - x), plotH);
    ctx.strokeStyle = COLORS.amber; ctx.lineWidth = 1.3; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, height - pad.bottom); ctx.stroke(); ctx.setLineDash([]);
  }
  drawAxes(ctx, width, height, pad, { xMin, xMax, yMin: range[0], yMax: range[1], xUnit: "s", yLabel: "ECG" });
}

function drawMaskBand(ctx, start, end, xMin, xMax, pad, width, height, fill, stroke, label, labelOffset) {
  const left = map(clamp(start, xMin, xMax), xMin, xMax, pad.left, width - pad.right);
  const right = map(clamp(end, xMin, xMax), xMin, xMax, pad.left, width - pad.right);
  if (right <= left) return;
  ctx.fillStyle = fill; ctx.fillRect(left, pad.top, right-left, height-pad.top-pad.bottom);
  ctx.strokeStyle = stroke; ctx.lineWidth = .8; ctx.setLineDash([2,2]);
  [left,right].forEach((x) => { ctx.beginPath(); ctx.moveTo(x,pad.top); ctx.lineTo(x,height-pad.bottom); ctx.stroke(); });
  ctx.setLineDash([]); ctx.fillStyle = stroke; ctx.font = "7px Segoe UI"; ctx.textAlign = "left"; ctx.fillText(label, left+2, pad.top+8+labelOffset);
}

function drawCalibration(calibrationCount) {
  const { ctx, width, height } = canvasContext($("#calibrationCanvas"));
  clearCanvas(ctx, width, height); drawChartBackground(ctx, width, height);
  if (!state.analysis) { drawEmptyMessage(ctx, width, height, "Waiting for analysis", "The first chronological QRS traces will accumulate here."); return; }
  const calibration = state.analysis.morphology_detail.patient_calibration;
  const display = calibration.visualization;
  const count = Math.min(calibrationCount, display.waveforms.length);
  if (!count || !display.waveform_time_ms.length) { drawEmptyMessage(ctx, width, height, "Waiting for the first complete QRS", "Automatic onset and offset estimates are required."); return; }
  const xs = display.waveform_time_ms, waveforms = display.waveforms.slice(0, count);
  const values = waveforms.flat().filter((value) => value != null), range = paddedRange(values, .13);
  const pad = { left: 46, right: 14, top: 18, bottom: 27 }, xMin = xs[0], xMax = xs.at(-1);
  const beats = calibration.beats.slice(0, count);
  const pre = beats.map((beat) => beat.onset_to_r_ms).sort((a,b)=>a-b);
  const post = beats.map((beat) => beat.r_to_offset_ms).sort((a,b)=>a-b);
  const runningPre = quantile(pre, .95), runningPost = quantile(post, .95);
  drawRelativeBand(ctx, -runningPre, runningPost, xMin, xMax, pad, width, height, "rgba(57,120,168,.09)", "rgba(57,120,168,.75)");
  waveforms.forEach((waveform, index) => {
    const alpha = .12 + .58 * (index + 1) / count;
    drawPolyline(ctx, xs, waveform, { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color: `rgba(193,111,31,${alpha})`, lineWidth: index === count-1 ? 1.55 : .8 });
    const beat = beats[index];
    drawWaveformBoundaryDot(ctx, xs, waveform, -beat.onset_to_r_ms, range, xMin, xMax, pad, width, height);
    drawWaveformBoundaryDot(ctx, xs, waveform, beat.r_to_offset_ms, range, xMin, xMax, pad, width, height);
  });
  const zeroX = map(0, xMin, xMax, pad.left, width-pad.right);
  ctx.strokeStyle = "rgba(41,65,77,.42)"; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(zeroX,pad.top); ctx.lineTo(zeroX,height-pad.bottom); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = COLORS.ink; ctx.font = "700 8px Segoe UI"; ctx.textAlign = "center"; ctx.fillText("R", zeroX, pad.top+9);
  drawAxes(ctx, width, height, pad, { xMin, xMax, yMin: range[0], yMax: range[1], xUnit: "ms", yLabel: "raw lead" });
}

function drawRelativeBand(ctx, startMs, endMs, xMin, xMax, pad, width, height, fill, stroke) {
  const left = map(clamp(startMs,xMin,xMax),xMin,xMax,pad.left,width-pad.right), right = map(clamp(endMs,xMin,xMax),xMin,xMax,pad.left,width-pad.right);
  ctx.fillStyle = fill; ctx.fillRect(left,pad.top,right-left,height-pad.top-pad.bottom);
  ctx.strokeStyle = stroke; ctx.lineWidth = .9; ctx.setLineDash([3,2]);
  [left,right].forEach((x)=>{ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,height-pad.bottom);ctx.stroke();}); ctx.setLineDash([]);
}

function drawWaveformBoundaryDot(ctx, xs, waveform, timeMs, range, xMin, xMax, pad, width, height) {
  const index = clamp(lowerBound(xs,timeMs),0,xs.length-1), value = waveform[index];
  if (value == null) return;
  const x = map(xs[index],xMin,xMax,pad.left,width-pad.right), y = map(value,range[0],range[1],height-pad.bottom,pad.top);
  ctx.fillStyle = COLORS.teal; ctx.beginPath(); ctx.arc(x,y,1.8,0,Math.PI*2); ctx.fill();
}

function drawMorphology(latestMorph) {
  const { ctx, width, height } = canvasContext($("#morphCanvas"));
  clearCanvas(ctx, width, height); drawChartBackground(ctx, width, height);
  const compact = document.body.classList.contains("one-screen");
  const pad = compact ? { left: 7, right: 6, top: 7, bottom: 10 } : { left: 42, right: 14, top: 13, bottom: 25 };
  if (!state.analysis || !latestMorph) {
    drawEmptyMessage(ctx, width, height, "Waiting for frozen patient calibration", "Five patient-window QRS crops will appear here."); return;
  }
  const data = state.analysis.morphology_beats;
  const start = latestMorph.start_beat_index, end = latestMorph.end_beat_index;
  const waveforms = data.waveforms.slice(start, end + 1).filter((row) => row.some((value) => value != null));
  if (!waveforms.length) { drawEmptyMessage(ctx, width, height, "QRS capture unavailable", "This window touches the selected segment edge."); return; }
  const values = waveforms.flat().filter((value)=>value!=null), range = paddedRange(values, .14);
  const xMin = data.waveform_time_ms[0], xMax = data.waveform_time_ms.at(-1);
  drawRelativeBand(ctx,xMin,xMax,xMin,xMax,pad,width,height,"rgba(57,120,168,.06)","rgba(57,120,168,.55)");
  const palette = ["rgba(57,120,168,.25)", "rgba(57,120,168,.38)", "rgba(57,120,168,.52)", "rgba(57,120,168,.68)", "#3978a8"];
  waveforms.forEach((waveform, index) => {
    drawPolyline(ctx, data.waveform_time_ms, waveform, { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color: palette[index], lineWidth: index === waveforms.length - 1 ? 1.65 : .9 });
  });
  const zeroX = map(0, xMin, xMax, pad.left, width - pad.right);
  ctx.strokeStyle = "rgba(57,120,168,.55)"; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(zeroX, pad.top); ctx.lineTo(zeroX, height - pad.bottom); ctx.stroke(); ctx.setLineDash([]);
  if (compact) {
    ctx.font="600 5.5px Segoe UI"; ctx.fillStyle="#3978a8"; ctx.textAlign="left";
    ctx.fillText("5 aligned QRS · R = 0 ms", pad.left + 2, pad.top + 6);
    ctx.fillStyle="#819097"; ctx.fillText(`${Math.round(xMin)} ms`, pad.left, height - 2);
    ctx.textAlign="center"; ctx.fillText("R", zeroX, height - 2);
    ctx.textAlign="right"; ctx.fillText(`+${Math.round(xMax)} ms`, width - pad.right, height - 2);
  } else {
    ctx.font="8px Segoe UI";ctx.textAlign="left";ctx.fillStyle="#3978a8";ctx.fillText("five patient-window QRS crops",pad.left+5,pad.top+10);
    drawAxes(ctx, width, height, pad, { xMin, xMax, yMin: range[0], yMax: range[1], xUnit: "ms", yLabel: "raw QRS" });
  }
}

function drawTemplate(latestTemplate) {
  const { ctx, width, height } = canvasContext($("#templateCanvas"));
  clearCanvas(ctx,width,height); drawChartBackground(ctx,width,height);
  if (!state.analysis) { drawEmptyMessage(ctx,width,height,"Waiting for calibration","Whole cardiac cycles will appear here."); return; }
  const lane = state.analysis.morphology_detail.patient_template;
  if (!lane.available) { drawEmptyMessage(ctx,width,height,"Whole-cycle lane unavailable",lane.status); return; }
  if (!latestTemplate) { drawEmptyMessage(ctx,width,height,"Waiting for a complete midpoint cycle","The first and last R anchors cannot own complete cycles."); return; }
  const pad={left:43,right:14,top:16,bottom:26}, xs=lane.waveform_phase, current=lane.waveforms[latestTemplate.waveform_row_index];
  const showTemplates = latestTemplate.evaluation_eligible;
  const templates = showTemplates ? lane.template_bank : [];
  const values=[...current,...templates.flat()].filter((value)=>value!=null), range=paddedRange(values,.14), xMin=-1,xMax=1;
  templates.forEach((row,index)=>drawPolyline(ctx,xs,row,{xMin,xMax,yMin:range[0],yMax:range[1],pad,width,height,color:`rgba(12,156,139,${.58-.12*index})`,lineWidth:2-index*.25}));
  drawPolyline(ctx,xs,current,{xMin,xMax,yMin:range[0],yMax:range[1],pad,width,height,color:latestTemplate.calibration_member?"rgba(214,135,50,.86)":"rgba(57,120,168,.9)",lineWidth:1.35});
  const zeroX=map(0,xMin,xMax,pad.left,width-pad.right);ctx.strokeStyle="rgba(41,65,77,.35)";ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(zeroX,pad.top);ctx.lineTo(zeroX,height-pad.bottom);ctx.stroke();ctx.setLineDash([]);
  ctx.font="8px Segoe UI";ctx.textAlign="left";ctx.fillStyle=latestTemplate.calibration_member?"#b96c20":"#3978a8";ctx.fillText(latestTemplate.calibration_member?"collecting calibration cycle":"current evaluation cycle",pad.left+5,pad.top+10);if(showTemplates){ctx.fillStyle="#0c897a";ctx.fillText("frozen template",pad.left+125,pad.top+10);}
  drawAxes(ctx,width,height,pad,{xMin,xMax,yMin:range[0],yMax:range[1],xUnit:"cycle phase",yLabel:"centered lead"});
}

function drawPqrst(row) {
  const { ctx,width,height }=canvasContext($("#pqrstCanvas")); clearCanvas(ctx,width,height); drawChartBackground(ctx,width,height);
  const strip=$("#landmarkStrip"); [...strip.children].forEach((item)=>item.className="");
  if(!state.analysis||!row){drawEmptyMessage(ctx,width,height,"Waiting for a delineated beat","Automatic P, QRS and T landmarks will appear here.");return;}
  const xMin=-260,xMax=520,start=row.time_s+xMin/1000,end=row.time_s+xMax/1000;
  const first=lowerBound(state.analysis.signal.times_s,start),last=upperBoundValues(state.analysis.signal.times_s,end),absTimes=state.analysis.signal.times_s.slice(first,last),values=state.analysis.signal.values.slice(first,last),xs=absTimes.map((time)=>(time-row.time_s)*1000);
  if(!values.length){drawEmptyMessage(ctx,width,height,"Beat touches the segment edge","Choose a later beat to see the full cycle.");return;}
  const pad={left:45,right:14,top:17,bottom:27},range=paddedRange(values,.15),rate=state.analysis.source.sampling_rate_hz,segmentStart=state.analysis.source.start_s;
  if(row.samples.qrs_onset!=null&&row.samples.qrs_offset!=null){const q1=(segmentStart+row.samples.qrs_onset/rate-row.time_s)*1000,q2=(segmentStart+row.samples.qrs_offset/rate-row.time_s)*1000;drawRelativeBand(ctx,q1,q2,xMin,xMax,pad,width,height,"rgba(12,156,139,.10)","rgba(12,156,139,.62)");}
  drawPolyline(ctx,xs,values,{xMin,xMax,yMin:range[0],yMax:range[1],pad,width,height,color:COLORS.teal,lineWidth:1.45});
  const landmarkNames=["p_onset","p_peak","p_offset","qrs_onset","r_peak","qrs_offset","t_onset","t_peak","t_offset"];
  const labels=["P on","P","P off","QRS on","R","QRS off","T on","T","T off"];
  landmarkNames.forEach((name,index)=>{const sample=name==="r_peak"?null:row.samples[name],relative=name==="r_peak"?0:(sample==null?null:(segmentStart+sample/rate-row.time_s)*1000);const pill=strip.children[index];pill.className=relative==null?"missing":"present";if(relative==null)return;const x=map(relative,xMin,xMax,pad.left,width-pad.right),signalIndex=clamp(lowerBound(xs,relative),0,xs.length-1),y=map(values[signalIndex],range[0],range[1],height-pad.bottom,pad.top);ctx.fillStyle=name.includes("qrs")||name==="r_peak"?"#0c8e80":"#a65d85";ctx.beginPath();ctx.arc(x,y,2.3,0,Math.PI*2);ctx.fill();ctx.font="700 7px Segoe UI";ctx.textAlign="center";ctx.fillText(labels[index],x,Math.max(pad.top+8,y-6));});
  drawAxes(ctx,width,height,pad,{xMin,xMax,yMin:range[0],yMax:range[1],xUnit:"ms from R",yLabel:"ECG"});
}

function drawHrv(rrRows) {
  const { ctx, width, height } = canvasContext($("#hrvCanvas"));
  clearCanvas(ctx, width, height); drawChartBackground(ctx, width, height);
  if (!rrRows.length) { drawEmptyMessage(ctx, width, height, "Waiting for two R peaks", "RR intervals will accumulate here."); return; }
  const visible = rrRows.slice(-100), times = visible.map((row) => row.time_s), values = visible.map((row) => row.rr_ms);
  const median = causalMedian(values, 7), pad = { left: 45, right: 14, top: 13, bottom: 25 };
  const range = paddedRange(values, .16, 20), xMin = times[0], xMax = Math.max(times.at(-1), xMin + 1);
  drawPolyline(ctx, times, median, { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color: COLORS.purple, lineWidth: 1.8 });
  visible.forEach((row) => {
    const x = map(row.time_s, xMin, xMax, pad.left, width - pad.right), y = map(row.rr_ms, range[0], range[1], height - pad.bottom, pad.top);
    ctx.fillStyle = row.supported ? "rgba(117,89,199,.72)" : COLORS.unsupported; ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  });
  drawAxes(ctx, width, height, pad, { xMin, xMax, yMin: range[0], yMax: range[1], xUnit: "s", yLabel: "RR ms" });
}

function drawPrsaBprsa(row) {
  const { ctx, width, height } = canvasContext($("#prsaCanvas"));
  clearCanvas(ctx, width, height); drawChartBackground(ctx, width, height);
  if (!state.analysis || !row?.defined) {
    const count = row?.window_beat_count || 0;
    drawEmptyMessage(ctx, width, height, count < 80 ? `Building 80-beat window · ${count} / 80` : "No complete anchor windows", count < 80 ? "The curve appears after 80 RR-defined beats." : "PRSA and BPRSA require usable ±5-second anchor neighborhoods.");
    return;
  }
  const xs = state.analysis.prsa_bprsa.curve_time_s;
  const prsa = row.prsa_curve_ms;
  const bprsa = row.bprsa_curve_ms;
  const pad = { left: 49, right: 15, top: 21, bottom: 27 };
  const range = paddedRange([...prsa, ...bprsa], .16, 1);
  const xMin = xs[0], xMax = xs.at(-1);
  drawPolyline(ctx, xs, prsa, { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color: COLORS.purple, lineWidth: 2.0 });
  drawPolyline(ctx, xs, bprsa, { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color: COLORS.teal, lineWidth: 1.8 });
  const anchorX = map(0, xMin, xMax, pad.left, width - pad.right);
  ctx.strokeStyle = "rgba(41,65,77,.46)"; ctx.setLineDash([4,3]); ctx.beginPath(); ctx.moveTo(anchorX, pad.top); ctx.lineTo(anchorX, height-pad.bottom); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = COLORS.purple; ctx.font = "700 8px Segoe UI"; ctx.textAlign = "left"; ctx.fillText(`PRSA · ${row.prsa_anchor_count} anchors`, pad.left + 6, pad.top + 9);
  ctx.fillStyle = COLORS.teal; ctx.fillText(`BPRSA · ${row.bprsa_anchor_count} anchors`, pad.left + 112, pad.top + 9);
  ctx.fillStyle = "#657780"; ctx.textAlign = "center"; ctx.fillText("AP", anchorX, pad.top - 5);
  drawAxes(ctx, width, height, pad, { xMin, xMax, yMin: range[0], yMax: range[1], xUnit: "s", yLabel: "averaged RR ms" });
}

function drawQuality(rows) {
  const { ctx, width, height } = canvasContext($("#qualityCanvas"));
  clearCanvas(ctx, width, height); drawChartBackground(ctx, width, height);
  if (!rows.length) { drawEmptyMessage(ctx, width, height, "Building the first causal 10-second window", "Future samples are not used; the first row appears at the 10-second endpoint."); return; }
  const visible = rows.slice(-20), times = visible.map((row) => row.time_s), pad = { left: 46, right: 15, top: 20, bottom: 27 };
  const xMin = times[0], xMax = Math.max(times.at(-1), xMin + 1), yMin = 0, yMax = 1;
  const series = [
    ["bassqi_clifford", COLORS.quality, "basSQI"],
    ["bsqi_li2008_jaccard_150ms", COLORS.purple, "bSQI"],
    ["template_corr_orphanidou", COLORS.amber, "beat corr"],
  ];
  series.forEach(([key, color]) => drawPolyline(ctx, times, visible.map((row) => row.available[key] ? row.values[key] : null), { xMin, xMax, yMin, yMax, pad, width, height, color, lineWidth: 1.9 }));
  const latest = visible.at(-1);
  series.forEach(([key, color, label], index) => {
    const value = latest.available[key] ? latest.values[key] : null;
    ctx.fillStyle = color; ctx.font = "700 8px Segoe UI"; ctx.textAlign = "left"; ctx.fillText(`${label} ${value == null ? "—" : formatNumber(value, 3)}`, pad.left + 5 + index * 92, pad.top + 9);
  });
  drawAxes(ctx, width, height, pad, { xMin, xMax, yMin, yMax, xUnit: "s", yLabel: "unitless context" });
}

function drawConduction(rows) {
  const { ctx, width, height } = canvasContext($("#conductionCanvas"));
  clearCanvas(ctx, width, height); drawChartBackground(ctx, width, height);
  if (!rows.length) { drawEmptyMessage(ctx, width, height, "Waiting for the first delineated beat", "Automatic landmark differences will appear in milliseconds."); return; }
  const visible = rows.slice(-80), times = visible.map((row) => row.r_peak_time_s), pad = { left: 49, right: 15, top: 20, bottom: 27 };
  const lines = [
    ["pr_interval_ms", COLORS.quality, "PR"],
    ["qrs_duration_ms", COLORS.amber, "QRS"],
    ["qt_interval_ms", COLORS.conduction, "QT"],
    ["jt_interval_ms", COLORS.purple, "JT"],
  ];
  const values = lines.flatMap(([key]) => visible.map((row) => row[key]).filter((value) => value != null));
  const range = paddedRange(values.length ? values : [0, 1], .12, 10), xMin = times[0], xMax = Math.max(times.at(-1), xMin + 1);
  lines.forEach(([key, color]) => drawPolyline(ctx, times, visible.map((row) => row[key]), { xMin, xMax, yMin: range[0], yMax: range[1], pad, width, height, color, lineWidth: 1.7 }));
  const latest = visible.at(-1);
  lines.forEach(([key, color, label], index) => { ctx.fillStyle = color; ctx.font = "700 8px Segoe UI"; ctx.textAlign = "left"; ctx.fillText(`${label} ${latest[key] == null ? "—" : `${formatNumber(latest[key], 1)} ms`}`, pad.left + 5 + index * 96, pad.top + 9); });
  drawAxes(ctx, width, height, pad, { xMin, xMax, yMin: range[0], yMax: range[1], xUnit: "s", yLabel: "automatic interval ms" });
}

function renderMatrix(allRows) {
  const allDefinitions = state.analysis?.feature_definitions || [];
  const definitions = allDefinitions.filter((definition)=>state.matrixGroup==="all"||definitionGroup(definition)===state.matrixGroup);
  const visible = allRows.slice(-8);
  const signature = `${state.matrixGroup}|${definitions.length}|${visible.length}|${visible.at(-1)?.time_s ?? "empty"}`;
  if (signature === state.matrixSignature) return;
  state.matrixSignature = signature;
  const table = $("#featureMatrix"), headRow = document.createElement("tr");
  headRow.innerHTML = '<th class="time-column">Time</th><th class="status-column">Data state</th>';
  definitions.forEach((definition) => {
    const th = document.createElement("th"); th.className = `${branchClass(definition.branch)}-column`;
    th.title = definition.meaning;
    th.innerHTML = `<span class="branch-label">${definition.branch}</span>${escapeHtml(definition.label)}<small>${escapeHtml(definition.unit)}</small>`;
    headRow.appendChild(th);
  });
  table.tHead.replaceChildren(headRow);
  const tbody = document.createElement("tbody");
  if (!visible.length) {
    const row = document.createElement("tr"); row.className = "matrix-empty-row";
    row.innerHTML = `<td colspan="${definitions.length + 2}"><div><strong>The matrix will assemble during playback</strong><span>Quality begins after 10 seconds, conduction builds 5/9/30-beat context, PRSA/BPRSA waits for 80 beats, and full HRV waits for 100 RR intervals.</span></div></td>`;
    tbody.appendChild(row); table.tBodies[0].replaceWith(tbody); return;
  }
  const scales = matrixScales(allRows, definitions);
  visible.forEach((entry) => {
    const tr = document.createElement("tr");
    const time = document.createElement("td"); time.className = "time-column"; time.textContent = formatClock(entry.time_s - state.analysis.source.start_s); tr.appendChild(time);
    const status = document.createElement("td"); status.className = "status-column";
    const stateName = entry.measurements_defined ? (entry.context_pass ? "complete" : "context") : "building";
    const stateText = entry.measurements_defined ? (entry.context_pass ? "Complete" : "Context flag") : "Building HRV";
    status.innerHTML = `<span class="data-state ${stateName}"><i></i>${stateText}</span>`; tr.appendChild(status);
    definitions.forEach((definition) => {
      const td = document.createElement("td"), value = entry.values[definition.key];
      if (value == null) { td.textContent = "—"; td.className = "matrix-value-missing"; }
      else {
        td.textContent = formatMatrixValue(value); td.title = `${definition.label}: ${value} ${definition.unit}`;
        const [minimum, maximum] = scales[definition.key], fraction = maximum > minimum ? clamp((value - minimum) / (maximum - minimum), 0, 1) : .45;
        td.style.background = heatColor(fraction, definitionGroup(definition)); td.style.color = fraction > .72 ? "#ffffff" : COLORS.ink;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.tBodies[0].replaceWith(tbody);
}

function definitionGroup(definition) {
  if(definition.branch.startsWith("Varon")) return "varon";
  if(definition.branch==="Template") return "template";
  if(definition.branch==="P–QRS–T") return "pqrst";
  if(definition.branch==="PRSA/BPRSA") return "prsa";
  if(definition.branch==="Signal quality") return "quality";
  if(definition.branch==="Conduction") return "conduction";
  return "hrv";
}

function branchClass(branch) { if(branch==="P–QRS–T")return "pqrst";return branch.toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-|-$/g,""); }

function matrixScales(rows, definitions) {
  const scales = {};
  definitions.forEach((definition) => {
    const values = rows.map((row) => row.values[definition.key]).filter((value) => value != null).sort((a,b) => a-b);
    if (!values.length) scales[definition.key] = [0, 1];
    else scales[definition.key] = [quantile(values, .05), quantile(values, .95)];
  });
  return scales;
}

function exportMatrixCsv() {
  if (!state.analysis) return;
  const definitions = state.analysis.feature_definitions;
  const header = ["time_s", "morphology_defined", "hrv_defined", "hrv_reliable", "prsa_bprsa_defined", "prsa_bprsa_reliable", "prsa_bprsa_computationally_usable", "prsa_bprsa_signal_quality_attached", "prsa_bprsa_model_eligible", "prsa_bprsa_role", "prsa_bprsa_affects_core_gate", "signal_quality_context_available", "signal_quality_artifact_label_produced", "signal_quality_affects_core_gate", "conduction_information_available", "conduction_final_matrix_eligible", "conduction_abnormality_label_produced", "conduction_affects_core_gate", "measurements_defined", "context_pass", ...definitions.map((item) => item.key)];
  const lines = [header.join(",")];
  state.analysis.fusion.forEach((row) => lines.push([
    row.time_s, row.morphology_defined, row.hrv_defined, row.hrv_reliable, row.prsa_bprsa_defined, row.prsa_bprsa_reliable, row.prsa_bprsa_computationally_usable, row.prsa_bprsa_signal_quality_attached, row.prsa_bprsa_model_eligible, row.prsa_bprsa_role, row.prsa_bprsa_affects_core_gate, row.signal_quality_context_available, row.signal_quality_artifact_label_produced, row.signal_quality_affects_core_gate, row.conduction_information_available, row.conduction_final_matrix_eligible, row.conduction_abnormality_label_produced, row.conduction_affects_core_gate, row.measurements_defined, row.context_pass,
    ...definitions.map((item) => row.values[item.key] ?? ""),
  ].join(",")));
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
  link.download = `${state.analysis.source.name.replace(/\.[^.]+$/, "")}_${state.analysis.track.key}_combined_features.csv`;
  document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(link.href);
}

function populateGlossary() {
  const grid = $("#glossaryGrid");
  if (!state.analysis) return;
  grid.replaceChildren(...state.analysis.feature_definitions.map((definition) => {
    const item = document.createElement("div"); item.className = "glossary-item";
    item.innerHTML = `<div class="glossary-symbol">${escapeHtml(definition.label)}<small>${escapeHtml(definition.branch)} · ${escapeHtml(definition.unit)} · ${escapeHtml(definition.role || "core_measurement")}</small></div><p>${escapeHtml(definition.meaning)}</p>`;
    return item;
  }));
}

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width)), height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, width, height };
}

function clearCanvas(ctx, width, height) { ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#fbfdfd"; ctx.fillRect(0, 0, width, height); }
function drawChartBackground(ctx, width, height) { ctx.strokeStyle = COLORS.minorGrid; ctx.lineWidth = 1; for (let x = 45; x < width; x += 42) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); } for (let y = 20; y < height; y += 31) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); } }
function drawEcgGrid(ctx, width, height) { ctx.lineWidth = 1; for (let x = 0; x <= width; x += 12) { ctx.strokeStyle = x % 60 === 0 ? "rgba(12,156,139,.11)" : "rgba(12,156,139,.045)"; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,height); ctx.stroke(); } for (let y = 0; y <= height; y += 12) { ctx.strokeStyle = y % 60 === 0 ? "rgba(12,156,139,.11)" : "rgba(12,156,139,.045)"; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(width,y); ctx.stroke(); } }

function drawPolyline(ctx, xs, ys, options) {
  const { xMin, xMax, yMin, yMax, pad, width, height, color, lineWidth } = options;
  ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.beginPath(); let started = false;
  for (let i = 0; i < xs.length; i += 1) {
    if (ys[i] == null || !Number.isFinite(ys[i])) { started = false; continue; }
    const x = map(xs[i], xMin, xMax, pad.left, width - pad.right), y = map(ys[i], yMin, yMax, height - pad.bottom, pad.top);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawAxes(ctx, width, height, pad, { xMin, xMax, yMin, yMax, xUnit, yLabel }) {
  ctx.strokeStyle = "#cfdbde"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, height-pad.bottom); ctx.lineTo(width-pad.right, height-pad.bottom); ctx.stroke();
  ctx.fillStyle = COLORS.muted; ctx.font = "9px Segoe UI"; ctx.textAlign = "center";
  for (let i = 0; i <= 4; i += 1) { const x = map(i,0,4,pad.left,width-pad.right), value = xMin + (xMax-xMin)*i/4; ctx.fillText(`${formatAxis(value)}${i === 4 ? ` ${xUnit}` : ""}`, x, height-9); }
  ctx.textAlign = "right"; ctx.fillText(formatAxis(yMax), pad.left-6, pad.top+3); ctx.fillText(formatAxis(yMin), pad.left-6, height-pad.bottom+3);
  ctx.save(); ctx.translate(11, height/2); ctx.rotate(-Math.PI/2); ctx.textAlign = "center"; ctx.fillStyle = "#91a0a6"; ctx.font = "8px Segoe UI"; ctx.fillText(yLabel, 0, 0); ctx.restore();
}

function drawEmptyMessage(ctx, width, height, title, subtitle) { const compact=document.body.classList.contains("one-screen");ctx.fillStyle="#617780";ctx.font=`600 ${compact?6.5:11}px Segoe UI`;ctx.textAlign="center";ctx.fillText(title,width/2,height/2-(compact?2:3));ctx.fillStyle="#98a5aa";ctx.font=`${compact?5.5:9}px Segoe UI`;ctx.fillText(subtitle,width/2,height/2+(compact?8:14)); }
function drawAllEmpty() { drawCalibration(0); drawMorphology(null); drawTemplate(null); drawPqrst(null); drawHrv([]); drawPrsaBprsa(null); drawQuality([]); drawConduction([]); }

function causalMedian(values, width) { return values.map((_, index) => { const slice = values.slice(Math.max(0,index-width+1),index+1).sort((a,b)=>a-b), mid=Math.floor(slice.length/2); return slice.length%2?slice[mid]:(slice[mid-1]+slice[mid])/2; }); }
function paddedRange(values, fraction = .1, floorPad = 1e-6) { let min=Math.min(...values), max=Math.max(...values); if (!Number.isFinite(min)||!Number.isFinite(max)) return [0,1]; const span=Math.max(max-min, floorPad), pad=span*fraction; return [min-pad,max+pad]; }
function map(value, a, b, c, d) { return b === a ? (c+d)/2 : c + (value-a)*(d-c)/(b-a); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function lowerBound(values, target) { let lo=0,hi=values.length; while(lo<hi){const mid=(lo+hi)>>1;if(values[mid]<target)lo=mid+1;else hi=mid;} return lo; }
function upperBoundValues(values, target) { let lo=0,hi=values.length; while(lo<hi){const mid=(lo+hi)>>1;if(values[mid]<=target)lo=mid+1;else hi=mid;} return lo; }
function upperBound(rows, target, key) { let lo=0,hi=rows.length; while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid][key]<=target)lo=mid+1;else hi=mid;} return lo; }
function quantile(sorted, q) { if (!sorted.length) return 0; const pos=(sorted.length-1)*q, base=Math.floor(pos), rest=pos-base; return sorted[base+1]===undefined?sorted[base]:sorted[base]+rest*(sorted[base+1]-sorted[base]); }
function heatColor(fraction, branch) { const palettes={varon:[[255,247,237],[202,113,31]],template:[[239,249,247],[12,143,127]],pqrst:[[252,242,247],[166,93,133]],hrv:[[245,242,252],[105,76,184]],prsa:[[239,249,247],[25,139,127]],quality:[[238,249,251],[22,137,162]],conduction:[[253,241,245],[180,91,120]]};const [start,end]=palettes[branch]||palettes.hrv,t=.12+.88*fraction; return `rgb(${start.map((value,index)=>Math.round(value+(end[index]-value)*t)).join(",")})`; }

function analysisEndS() { return state.analysis.source.start_s + state.analysis.source.duration_s; }
function formatNumber(value, digits=1) { return Number(value).toLocaleString(undefined,{maximumFractionDigits:digits}); }
function compactNumber(value) { const abs=Math.abs(value); if(abs>=1000)return value.toExponential(2); if(abs>=10)return formatNumber(value,1); if(abs>=1)return formatNumber(value,2); return value.toExponential(2); }
function formatMatrixValue(value) { const abs=Math.abs(value); if(abs===0)return "0"; if(abs>=1000||abs<.01)return value.toExponential(1); if(abs>=100)return formatNumber(value,0); if(abs>=10)return formatNumber(value,1); return formatNumber(value,2); }
function formatAxis(value) { const abs=Math.abs(value); if(abs>=1000)return `${(value/1000).toFixed(1)}k`; if(abs<.01&&abs>0)return value.toExponential(1); return formatNumber(value, abs<10?1:0); }
function formatClock(seconds) { const value=Math.max(0,Math.round(seconds)); return `${Math.floor(value/60)}:${String(value%60).padStart(2,"0")}`; }
function formatDuration(seconds) { if(seconds>=3600)return `${(seconds/3600).toFixed(1)} h`; if(seconds>=60)return `${Math.round(seconds/60)} min`; return `${Math.round(seconds)} s`; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g,(character)=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[character])); }
function setStatus(element, kind, text) { element.className=`status-pill ${kind}`; element.innerHTML=`<span></span>${escapeHtml(text)}`; }

function showLoading(title, message) { $("#loadingTitle").textContent=title; $("#loadingMessage").textContent=message; $("#loadingOverlay").hidden=false; }
function hideLoading() { $("#loadingOverlay").hidden=true; }
function showToast(title, message, error=false) { const toast=$("#toast"); clearTimeout(state.toastTimer); toast.classList.toggle("error",error); $("#toastIcon").textContent=error?"!":"✓"; $("#toastTitle").textContent=title; $("#toastMessage").textContent=message; toast.classList.add("show"); state.toastTimer=setTimeout(()=>toast.classList.remove("show"),5000); }
function showError(error) { console.error(error); showToast("Could not continue", error.message || String(error), true); }
