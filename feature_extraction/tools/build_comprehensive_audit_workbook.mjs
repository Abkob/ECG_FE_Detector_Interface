import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const sourceDir = path.resolve(
  process.argv[2] ?? "outputs/comprehensive_branch_matrix_v1",
);
const outputPath = path.resolve(
  process.argv[3] ?? "outputs/comprehensive_ecg_branch_dataset_audit.xlsx",
);
const previewDir = path.resolve(
  process.argv[4] ?? ".artifact_previews/comprehensive_ecg_branch_dataset_audit",
);

const report = JSON.parse(
  await fs.readFile(path.join(sourceDir, "validation_report.json"), "utf8"),
);
const modelManifest = JSON.parse(
  await fs.readFile(path.join(sourceDir, "model_manifest.json"), "utf8"),
);

const palette = {
  navy: "#16324F",
  teal: "#0E7490",
  blue: "#2563EB",
  pale: "#E8F1F5",
  paleBlue: "#EAF2FF",
  green: "#15803D",
  paleGreen: "#DCFCE7",
  amber: "#D97706",
  paleAmber: "#FEF3C7",
  ink: "#1F2937",
  muted: "#64748B",
  border: "#CBD5E1",
  white: "#FFFFFF",
};

const importedMatrices = [];
for (const [sheetName, fileName] of [
  ["Feature Dictionary", "feature_dictionary.csv"],
  ["Label Dictionary", "label_dictionary.csv"],
  ["Branch Combinations", "branch_combinations.csv"],
]) {
  const csvText = await fs.readFile(path.join(sourceDir, fileName), "utf8");
  const importedWorkbook = await Workbook.fromCSV(csvText, { sheetName });
  const importedSheet = importedWorkbook.worksheets.getItem(sheetName);
  importedMatrices.push([sheetName, importedSheet.getUsedRange().values]);
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const coverage = workbook.worksheets.add("Dataset Coverage");
const labels = workbook.worksheets.add("Label Counts");
const tasks = workbook.worksheets.add("Model Tasks");
const qa = workbook.worksheets.add("QA & Files");
for (const [sheetName, matrix] of importedMatrices) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.getRange("A1").write(matrix);
}

function formatTitle(sheet, title, subtitle, lastColumn = "H") {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: palette.navy,
    font: { bold: true, color: palette.white, size: 18 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 30;
  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: palette.pale,
    font: { color: palette.ink, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 34;
}

function formatHeader(range) {
  range.format = {
    fill: palette.teal,
    font: { bold: true, color: palette.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: palette.border },
  };
  range.format.rowHeight = 28;
}

function formatBody(range) {
  range.format = {
    font: { color: palette.ink, size: 10 },
    wrapText: true,
    verticalAlignment: "top",
    borders: { preset: "all", style: "thin", color: palette.border },
  };
}

// Dataset coverage is the source table for summary formulas and the chart.
const datasetNotes = {
  butqdb: ["Quality", "18 long records; class-targeted annotated windows"],
  ludb: ["Diagnoses + landmarks", "All 200 ten-second records"],
  mitdb: ["Beat symbols", "All 48 records; rarity-weighted 180-second windows"],
  nstdb: ["Noise/SNR", "All 12 electrode-motion SNR conditions"],
  qtdb: ["Manual landmarks", "12 branch-qualified MLII q1c records"],
  seizure_edf: ["Seizure intervals", "All 9 configured EDFs; events plus controls"],
};
const coverageRows = Object.entries(report.rows_by_dataset).map(
  ([datasetKey, rowCount]) => [
    datasetKey,
    report.records_present_by_dataset[datasetKey],
    rowCount,
    null,
    datasetNotes[datasetKey][0],
    datasetNotes[datasetKey][1],
  ],
);
formatTitle(
  coverage,
  "Dataset coverage",
  "Strict record-level coverage after extraction; no planned record is silently omitted.",
  "F",
);
coverage.getRange("A4:F4").values = [[
  "Dataset",
  "Records represented",
  "Observation rows",
  "Share of rows",
  "Primary label namespace",
  "Validation scope",
]];
coverage.getRange(`A5:F${4 + coverageRows.length}`).values = coverageRows;
const coverageLast = 4 + coverageRows.length;
coverage.getRange("D5").formulas = [[`=C5/SUM($C$5:$C$${coverageLast})`]];
coverage.getRange(`D5:D${coverageLast}`).fillDown();
coverage.getRange(`A${coverageLast + 1}:F${coverageLast + 1}`).values = [[
  "TOTAL",
  null,
  null,
  1,
  "Separate targets",
  "All planned sources represented",
]];
coverage.getRange(`B${coverageLast + 1}`).formulas = [[`=SUM(B5:B${coverageLast})`]];
coverage.getRange(`C${coverageLast + 1}`).formulas = [[`=SUM(C5:C${coverageLast})`]];
formatHeader(coverage.getRange("A4:F4"));
formatBody(coverage.getRange(`A5:F${coverageLast + 1}`));
coverage.getRange(`A${coverageLast + 1}:F${coverageLast + 1}`).format = {
  fill: palette.paleBlue,
  font: { bold: true, color: palette.navy },
  borders: { preset: "all", style: "thin", color: palette.border },
};
coverage.getRange(`C5:C${coverageLast + 1}`).format.numberFormat = "#,##0";
coverage.getRange(`D5:D${coverageLast + 1}`).format.numberFormat = "0.0%";
coverage.tables.add(`A4:F${coverageLast}`, true, "DatasetCoverageTable");
coverage.freezePanes.freezeRows(4);
coverage.getRange("A1:A20").format.columnWidth = 18;
coverage.getRange("B1:C20").format.columnWidth = 16;
coverage.getRange("D1:D20").format.columnWidth = 14;
coverage.getRange("E1:E20").format.columnWidth = 22;
coverage.getRange("F1:F20").format.columnWidth = 48;
coverage.getRange(`A4:F${coverageLast + 1}`).format.autofitRows();

// Label counts preserve each namespace rather than forcing one universal y.
const labelCountRows = [];
for (const [family, counts] of Object.entries(report.label_counts)) {
  for (const [value, count] of Object.entries(counts)) {
    labelCountRows.push([family, value, count, null]);
  }
}
labels.showGridLines = false;
labels.getRange("A1:D1").values = [[
  "Label family",
  "Exact/derived value",
  "Row count",
  "Share within family",
]];
labels.getRange(`A2:D${labelCountRows.length + 1}`).values = labelCountRows;
const labelLast = labelCountRows.length + 1;
labels.getRange("D2").formulas = [[
  `=C2/SUMIF($A$2:$A$${labelLast},A2,$C$2:$C$${labelLast})`,
]];
labels.getRange(`D2:D${labelLast}`).fillDown();
formatHeader(labels.getRange("A1:D1"));
formatBody(labels.getRange(`A2:D${labelLast}`));
labels.getRange(`C2:C${labelLast}`).format.numberFormat = "#,##0";
labels.getRange(`D2:D${labelLast}`).format.numberFormat = "0.0%";
labels.tables.add(`A1:D${labelLast}`, true, "LabelCountsTable");
labels.freezePanes.freezeRows(1);
labels.getRange("A1:A80").format.columnWidth = 30;
labels.getRange("B1:B80").format.columnWidth = 22;
labels.getRange("C1:D80").format.columnWidth = 17;
labels.getRange(`A1:D${labelLast}`).format.autofitRows();

// Model task manifest.
const taskRows = Object.entries(modelManifest.tasks).map(([taskName, item]) => [
  taskName,
  item.rows,
  item.targets.join("\n"),
  item.group_column,
  item.fold_column,
  item.path,
  "Fit each branch subset separately; impute inside training folds.",
]);
tasks.showGridLines = false;
tasks.getRange("A1:G1").values = [[
  "Task",
  "Eligible rows",
  "Target column(s)",
  "Group column",
  "Fold column",
  "CSV view",
  "Training note",
]];
tasks.getRange(`A2:G${taskRows.length + 1}`).values = taskRows;
formatHeader(tasks.getRange("A1:G1"));
formatBody(tasks.getRange(`A2:G${taskRows.length + 1}`));
tasks.getRange(`B2:B${taskRows.length + 1}`).format.numberFormat = "#,##0";
tasks.tables.add(`A1:G${taskRows.length + 1}`, true, "ModelTasksTable");
tasks.freezePanes.freezeRows(1);
tasks.getRange("A1:A20").format.columnWidth = 28;
tasks.getRange("B1:B20").format.columnWidth = 14;
tasks.getRange("C1:C20").format.columnWidth = 40;
tasks.getRange("D1:E20").format.columnWidth = 22;
tasks.getRange("F1:F20").format.columnWidth = 36;
tasks.getRange("G1:G20").format.columnWidth = 54;
tasks.getRange(`A1:G${taskRows.length + 1}`).format.autofitRows();

// QA checklist and file map.
formatTitle(
  qa,
  "QA, modeling contract, and file map",
  "Green checks are required before classification or post-hoc cluster interpretation.",
  "F",
);
qa.getRange("A4:C4").values = [["Check", "Result", "Interpretation"]];
const qaRows = [
  ["Validation status", report.validation_passed ? "PASS" : "FAIL", "Strict builder validation"],
  ["Failed segments", report.failed_segments, "Must be zero"],
  ["Missing planned records", report.missing_records.length, "Must be zero"],
  ["Lineage groups crossing folds", report.lineage_groups_crossing_folds, "Must be zero"],
  ["Feature columns", report.feature_columns, "Exactly the maintained B1-B4 measurements"],
  ["Label columns", report.label_columns, "Kept outside X_features.csv"],
  ["Branch combinations", report.branch_combinations, "All non-empty subsets of four branches"],
  ["Missing-value policy", "PRESERVED", "Impute inside each training fold; never replace globally with zero"],
  ["Cluster label policy", "POST-HOC ONLY", "Fit clusters without labels, then join y_labels.csv by row_id"],
];
qa.getRange(`A5:C${qaRows.length + 4}`).values = qaRows;
formatHeader(qa.getRange("A4:C4"));
formatBody(qa.getRange(`A5:C${qaRows.length + 4}`));
qa.getRange("B5:B13").conditionalFormats.add("containsText", {
  text: "PASS",
  format: { fill: palette.paleGreen, font: { bold: true, color: palette.green } },
});
qa.getRange("A16:C16").values = [["File", "Purpose", "Location"]];
const fileRows = [
  ["X_features.csv", "Label-free row_id + 52 model inputs", report.artifacts.features],
  ["y_labels.csv", "Keyed original and derived labels", report.artifacts.labels],
  ["observations_with_labels.csv", "Joined audit view; not direct model input", report.artifacts.observations],
  ["row_provenance_and_qc.csv", "Source, split, availability, and QC fields", report.artifacts.provenance],
  ["branch_combinations.csv", "All 15 B1-B4 feature subsets", report.artifacts.branch_combinations],
  ["model_manifest.json", "Targets, task paths, grouping rule", report.artifacts.model_manifest],
  ["validation_report.json", "Strict build metrics", path.join(sourceDir, "validation_report.json")],
];
qa.getRange(`A17:C${fileRows.length + 16}`).values = fileRows;
formatHeader(qa.getRange("A16:C16"));
formatBody(qa.getRange(`A17:C${fileRows.length + 16}`));
qa.getRange("E4:F4").values = [["Reference", "URL / note"]];
qa.getRange("E5:F10").values = [
  ["BUT-QDB", "https://physionet.org/content/butqdb/1.0.0/"],
  ["LUDB", "https://physionet.org/content/ludb/1.0.1/"],
  ["QTDB", "https://physionet.org/content/qtdb/1.0.0/"],
  ["Local build README", path.join(sourceDir, "README.md")],
  ["Join rule", "row_id is the only feature-label join key"],
  ["Split rule", "lineage_group_id / suggested_cv_fold; never random rows"],
];
formatHeader(qa.getRange("E4:F4"));
formatBody(qa.getRange("E5:F10"));
qa.freezePanes.freezeRows(4);
qa.getRange("A1:A30").format.columnWidth = 30;
qa.getRange("B1:B30").format.columnWidth = 24;
qa.getRange("C1:C30").format.columnWidth = 62;
qa.getRange("D1:D30").format.columnWidth = 3;
qa.getRange("E1:E30").format.columnWidth = 24;
qa.getRange("F1:F30").format.columnWidth = 58;
qa.getRange("A4:F23").format.autofitRows();

// Summary sheet links to the detailed sheets with formulas.
formatTitle(
  summary,
  "Comprehensive ECG branch-validation dataset",
  "Label-preserving audit of B1 peak/HRV, B2 morphology, B3 conduction/repolarization, and B4 signal quality.",
  "M",
);
summary.getRange("A4:C4").values = [["Metric", "Value", "Why it matters"]];
const summaryRows = [
  ["Observations", null, "Beat-aligned rows across all validation windows"],
  ["Source records", null, "Every planned record is represented"],
  ["Feature columns", report.feature_columns, "Model inputs only; labels excluded"],
  ["Label/provenance columns", report.label_columns, "Separate keyed y view"],
  ["Branch combinations", report.branch_combinations, "All non-empty B1-B4 ablations"],
  ["Planned segments", report.planned_segments, "Label-aware event, class, and control windows"],
  ["Failed segments", report.failed_segments, "Strictly zero"],
  ["Missing records", report.missing_records.length, "Strictly zero"],
];
summary.getRange("A5:C12").values = summaryRows;
summary.getRange("B5").formulas = [[`='Dataset Coverage'!C${coverageLast + 1}`]];
summary.getRange("B6").formulas = [[`='Dataset Coverage'!B${coverageLast + 1}`]];
formatHeader(summary.getRange("A4:C4"));
formatBody(summary.getRange("A5:C12"));
summary.getRange("B5:B12").format.numberFormat = "#,##0";
summary.getRange("A14:C14").values = [["Modeling rule", "Required action", "Reason"]];
summary.getRange("A15:C19").values = [
  ["Features", "Train from X_features.csv", "Prevents label and provenance leakage"],
  ["Labels", "Join y_labels.csv by row_id", "Keeps exact source labels aligned after filtering"],
  ["Splits", "Use lineage_group_id / suggested_cv_fold", "Keeps related windows and source derivatives together"],
  ["Missing values", "Impute within training folds", "Missing means not measurable, not zero"],
  ["Clustering", "Use labels only after fitting", "Preserves unsupervised evaluation integrity"],
];
formatHeader(summary.getRange("A14:C14"));
formatBody(summary.getRange("A15:C19"));
summary.getRange("A21:C21").values = [["Validation", report.validation_passed ? "PASS" : "FAIL", "See QA & Files"]];
summary.getRange("A21:C21").format = {
  fill: report.validation_passed ? palette.paleGreen : palette.paleAmber,
  font: { bold: true, color: report.validation_passed ? palette.green : palette.amber },
  borders: { preset: "all", style: "medium", color: palette.border },
};
summary.getRange("A1:A24").format.columnWidth = 28;
summary.getRange("B1:B24").format.columnWidth = 22;
summary.getRange("C1:C24").format.columnWidth = 58;
summary.getRange("A4:C21").format.autofitRows();
summary.freezePanes.freezeRows(4);

const chart = summary.charts.add("bar", coverage.getRange(`A4:C${coverageLast}`));
chart.title = "Observation rows by dataset";
chart.hasLegend = false;
chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
chart.yAxis = { numberFormatCode: "#,##0" };
chart.setPosition("E4", "M19");

// Style imported dictionaries and branch manifest.
for (const [sheetName, lastColumn, widths] of [
  ["Feature Dictionary", "J", [8, 30, 14, 10, 42, 20, 14, 22, 68, 14]],
  ["Label Dictionary", "D", [38, 24, 34, 74]],
  ["Branch Combinations", "F", [24, 12, 24, 54, 14, 90]],
]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const matrix = importedMatrices.find(([name]) => name === sheetName)[1];
  const rowCount = matrix.length;
  sheet.showGridLines = false;
  const used = sheet.getUsedRange();
  formatBody(used);
  formatHeader(used.getRow(0));
  sheet.freezePanes.freezeRows(1);
  for (let index = 0; index < widths.length; index += 1) {
    sheet.getRangeByIndexes(0, index, rowCount, 1).format.columnWidth = widths[index];
  }
  if (sheetName === "Feature Dictionary") {
    sheet.getRange("A2:A60").format.numberFormat = "0";
  }
  if (sheetName === "Branch Combinations") {
    sheet.getRange("B2:B20").format.numberFormat = "0";
    sheet.getRange("E2:E20").format.numberFormat = "0";
  }
  sheet.getRange(`A1:${lastColumn}${rowCount}`).format.wrapText = true;
  sheet.getRange(`A1:${lastColumn}${rowCount}`).format.autofitRows();
}

// Keep renderer edge-crop artifacts outside the real content area.
function addWhiteGutter(sheetName, gutterColumnIndex, gutterRowIndex) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const gutterFormat = { fill: palette.white, font: { color: palette.white } };
  const right = sheet.getRangeByIndexes(
    0,
    gutterColumnIndex,
    gutterRowIndex + 1,
    1,
  );
  right.values = Array.from({ length: gutterRowIndex + 1 }, () => [" "]);
  right.format = gutterFormat;
  right.format.columnWidth = 2;
  const bottom = sheet.getRangeByIndexes(
    gutterRowIndex,
    0,
    1,
    gutterColumnIndex + 1,
  );
  bottom.values = [Array.from({ length: gutterColumnIndex + 1 }, () => " ")];
  bottom.format = gutterFormat;
  bottom.format.rowHeight = 6;
}

addWhiteGutter("Summary", 13, 21);
addWhiteGutter("Dataset Coverage", 6, coverageLast + 1);
addWhiteGutter("Label Counts", 4, labelLast);
addWhiteGutter("Model Tasks", 7, taskRows.length + 1);
addWhiteGutter("Feature Dictionary", 10, importedMatrices[0][1].length);
addWhiteGutter("Label Dictionary", 4, importedMatrices[1][1].length);
addWhiteGutter("Branch Combinations", 6, importedMatrices[2][1].length);
addWhiteGutter("QA & Files", 6, 23);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 12000,
  tableMaxRows: 8,
  tableMaxCols: 8,
  options: { maxResults: 100 },
});
console.log(inspection.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
  maxChars: 6000,
});
console.log(formulaErrors.ndjson);

for (const sheetName of [
  "Summary",
  "Dataset Coverage",
  "Label Counts",
  "Model Tasks",
  "Feature Dictionary",
  "Label Dictionary",
  "Branch Combinations",
  "QA & Files",
]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 0.9,
    format: "png",
  });
  const safeName = sheetName.toLowerCase().replaceAll(/[^a-z0-9]+/g, "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
console.log(JSON.stringify({ outputPath, previewDir }));
