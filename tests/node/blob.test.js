import test from "node:test";
import assert from "node:assert/strict";

import { validateBlobPath, validateUploadRequest } from "../../api/blob.js";

test("accepts supported ECG uploads within the hosted size limit", () => {
  assert.deepEqual(
    validateUploadRequest({ session: "browser-123", filename: "patient 1.EDF", bytes: 4096 }),
    { session: "browser-123", filename: "patient_1.EDF", bytes: 4096 },
  );
});

test("rejects unsupported or oversized uploads", () => {
  assert.throws(
    () => validateUploadRequest({ session: "browser-123", filename: "report.pdf", bytes: 10 }),
    /Unsupported uploaded file type/,
  );
  assert.throws(
    () => validateUploadRequest({ session: "browser-123", filename: "signal.edf", bytes: 17 * 1024 * 1024 }),
    /16 MB or smaller/,
  );
});

test("allows cleanup only inside the same browser-session prefix", () => {
  assert.equal(
    validateBlobPath("browser-123", "ecg-uploads/browser-123/id-signal.dat"),
    "ecg-uploads/browser-123/id-signal.dat",
  );
  assert.throws(
    () => validateBlobPath("browser-123", "ecg-uploads/another-session/id-signal.dat"),
    /does not belong/,
  );
});
