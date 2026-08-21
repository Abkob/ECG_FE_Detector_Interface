import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";


const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const sourceRoot = join(projectRoot, "feature_extraction", "src", "ecg_cascade", "web_demo");
const outputRoot = join(projectRoot, "public");

await mkdir(outputRoot, { recursive: true });
await Promise.all(
  ["index.html", "app.js", "styles.css"].map((name) =>
    copyFile(join(sourceRoot, name), join(outputRoot, name)),
  ),
);

process.stdout.write("Prepared the ECG Cascade static frontend for Vercel.\n");
