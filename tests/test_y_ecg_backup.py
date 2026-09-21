"""Behavioral tests for selective copying and non-destructive repeat refreshes."""
import importlib.util
import json
import shutil
import unittest
import uuid
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("backup", Path(__file__).resolve().parents[1] / "tools/sync_y_ecg_backup.py")
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


class BackupTests(unittest.TestCase):
    def setUp(self):
        test_parent = Path(__file__).resolve().parents[1] / "tmp" / "backup_tests"
        test_root = test_parent / uuid.uuid4().hex
        test_root.mkdir(parents=True)
        def cleanup():
            if not test_root.resolve().is_relative_to(test_parent.resolve()):
                raise ValueError("Test cleanup escaped its workspace")
            shutil.rmtree(test_root)
        self.addCleanup(cleanup)
        self.root = test_root / "source"
        self.dest = test_root / "backup"
        self.root.mkdir()
        self.put("README.md", "# Example\n")
        self.put("requirements.txt", "numpy\n")
        self.put("feature_extraction/src/ecg_cascade/rr_hrv.py", "value = 1\n")

    def put(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def sync(self, note="Initial fixture"):
        return backup.sync(self.root, self.dest, note)

    def test_selective_copy_and_verified_baseline(self):
        self.put(".env.local", "private-token\n")
        self.put("node_modules/example/main.js", "installed dependency")
        self.put("feature_extraction/outputs/run/console.txt", "dump")
        self.put("feature_extraction/reports/case/rendered/page-1.png", "preview")
        self.put("feature_extraction/reports/case/main.synctex.gz", "sidecar")
        self.put("feature_extraction/outputs/final/REPORT.md", "Conclusion")
        self.put("docs/research/literature-review.pdf", "%PDF-1.4\nretained research")
        self.put("feature_extraction/src/ecg_cascade/web_demo/audit_results.json", '{"ok":true}')
        result = self.sync()
        manifest = backup.load_manifest(self.dest)
        sources = {r.get("source") for r in manifest["files"].values()}
        self.assertTrue({"requirements.txt", "feature_extraction/outputs/final/REPORT.md"} <= sources)
        self.assertIn("feature_extraction/src/ecg_cascade/web_demo/audit_results.json", sources)
        self.assertIn("docs/research/literature-review.pdf", sources)
        self.assertFalse({".env.local", "node_modules/example/main.js", "feature_extraction/outputs/run/console.txt"} & sources)
        self.assertNotIn("feature_extraction/reports/case/rendered/page-1.png", sources)
        self.assertNotIn("feature_extraction/reports/case/main.synctex.gz", sources)
        self.assertEqual(result["verified_files"], backup.verify(self.dest, manifest))

    def test_repeat_noop_change_delete_and_unmanaged_preservation(self):
        self.sync()
        before = (self.dest / backup.MANIFEST).read_bytes()
        self.assertEqual(self.sync()["status"], "unchanged; verified")
        self.assertEqual(before, (self.dest / backup.MANIFEST).read_bytes())
        unmanaged = self.dest / "personal.md"
        unmanaged.write_text("keep me")
        self.put("feature_extraction/src/ecg_cascade/rr_hrv.py", "value = 2\n")
        result = self.sync("Change feature")
        self.assertGreaterEqual(result["changed"], 2)
        (self.root / "feature_extraction/src/ecg_cascade/rr_hrv.py").unlink()
        self.assertGreaterEqual(self.sync("Remove feature")["removed"], 2)
        self.assertEqual(unmanaged.read_text(), "keep me")
        self.assertFalse((self.dest / "00_project/code/feature_extraction/src/ecg_cascade/rr_hrv.py").exists())

    def test_manual_edit_stops_before_mutation(self):
        self.sync()
        modified = self.dest / "00_project/code/README.md"
        modified.write_text("manual edit")
        before = (self.dest / backup.MANIFEST).read_bytes()
        self.put("README.md", "source change")
        with self.assertRaisesRegex(ValueError, "Refusing"):
            self.sync("New change")
        self.assertEqual(modified.read_text(), "manual edit")
        self.assertEqual(before, (self.dest / backup.MANIFEST).read_bytes())

    def test_source_drift_and_corrupted_backup_detected(self):
        self.sync()
        manifest = backup.load_manifest(self.dest)
        self.put("README.md", "changed after backup")
        with self.assertRaisesRegex(ValueError, "sources changed"):
            backup.verify_sources(self.root, manifest)
        (self.dest / "00_project/code/README.md").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "verification failed"):
            backup.verify(self.dest, manifest)

    def test_duplicate_documents_and_notebook_outputs(self):
        self.put("feature_extraction/reports/rr_hrv.pdf", "%PDF-1.4\nexample")
        self.put("output/evidence/artifacts/rr_hrv_copy.pdf", "%PDF-1.4\nexample")
        path = self.put("feature_extraction/notebooks/rr_hrv.ipynb", json.dumps({"cells": [
            {"cell_type": "code", "source": ["print(1)"], "outputs": [{"text": "large output"}], "execution_count": 4}],
            "metadata": {"widgets": {"state": "large"}}, "nbformat": 4}))
        original = path.read_bytes()
        self.sync()
        manifest = backup.load_manifest(self.dest)
        self.assertEqual(len(manifest["document_aliases"]), 1)
        self.assertEqual(sum(p.endswith(".pdf") for p in manifest["files"]), 1)
        notebook = json.loads((self.dest / "00_project/code/feature_extraction/notebooks/rr_hrv.ipynb").read_text())
        self.assertEqual(notebook["cells"][0]["outputs"], [])
        self.assertEqual(path.read_bytes(), original)

    def test_dry_run_is_read_only_and_path_escape_refused(self):
        backup.sync(self.root, self.dest, None, dry_run=True)
        self.assertFalse(self.dest.exists())
        for bad in ("../outside", "/absolute", "C:/outside", "nested/../../outside", "a\\b"):
            with self.assertRaises(ValueError):
                backup.safe_path(self.dest, bad)
        with self.assertRaises(ValueError):
            backup.sync(self.root, self.root, "unsafe")

    def test_manual_update_index_edit_is_preserved(self):
        self.sync()
        path = self.dest / "01_rpeak_rr_hrv/UPDATES.md"
        path.write_text("personal history")
        with self.assertRaisesRegex(ValueError, "Manually edited"):
            self.sync("New decision")
        self.assertEqual(path.read_text(), "personal history")


if __name__ == "__main__":
    unittest.main()
