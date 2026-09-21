import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";


const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const sourceRoot = join(projectRoot, "feature_extraction", "src", "ecg_cascade", "web_demo");
const outputRoot = join(projectRoot, "public");
const auditArtifactRoot = sourceRoot;

await mkdir(outputRoot, { recursive: true });
await Promise.all(
  [
    "index.html",
    "app.js",
    "styles.css",
    "models.html",
    "models.js",
    "benchmark_results.json",
    "audit.html",
    "audit.js",
    "audit_results.json",
  ].map((name) =>
    copyFile(join(sourceRoot, name), join(outputRoot, name)),
  ),
);

await Promise.all(
  [
    "selected_case_predictions.csv.gz",
    "weak_case_explanations.csv.gz",
    "record_metrics.csv",
    "patient_metrics.csv",
    "group_oof_results.csv",
    "split_manifest.csv",
  ].map((name) => copyFile(join(auditArtifactRoot, name), join(outputRoot, name))),
);

process.stdout.write("Prepared the ECG Cascade static frontend for Vercel.\n");
