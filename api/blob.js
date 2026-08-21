import { del, issueSignedToken, presignUrl } from "@vercel/blob";
import { randomUUID } from "node:crypto";
import path from "node:path";

const ALLOWED_SUFFIXES = new Set([".edf", ".hea", ".dat", ".atr", ".csv", ".txt"]);
const SESSION_PATTERN = /^[A-Za-z0-9_-]{1,80}$/;
const MAX_UPLOAD_BYTES = 16 * 1024 * 1024;
const UPLOAD_URL_LIFETIME_MS = 10 * 60 * 1000;
const SOURCE_URL_LIFETIME_MS = 60 * 60 * 1000;

export function validateUploadRequest(input) {
  const session = String(input?.session || "");
  if (!SESSION_PATTERN.test(session)) throw new Error("Invalid browser session identifier.");

  const originalName = path.basename(String(input?.filename || ""));
  const filename = originalName
    .normalize("NFKC")
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^\.+/, "")
    .slice(-140);
  if (!filename) throw new Error("The uploaded file needs a valid filename.");
  const suffix = path.extname(filename).toLowerCase();
  if (!ALLOWED_SUFFIXES.has(suffix)) {
    throw new Error(`Unsupported uploaded file type: ${suffix || "(none)"}.`);
  }

  const bytes = Number(input?.bytes);
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new Error("The uploaded file size is invalid.");
  }
  if (bytes > MAX_UPLOAD_BYTES) {
    throw new Error("Each hosted ECG file must be 16 MB or smaller.");
  }
  return { session, filename, bytes };
}

export function validateBlobPath(session, pathname) {
  if (!SESSION_PATTERN.test(String(session || ""))) {
    throw new Error("Invalid browser session identifier.");
  }
  const value = String(pathname || "");
  if (!value.startsWith(`ecg-uploads/${session}/`) || value.includes("..")) {
    throw new Error("The private ECG object does not belong to this browser session.");
  }
  return value;
}

async function readJson(request) {
  if (request.body && typeof request.body === "object" && !Buffer.isBuffer(request.body)) {
    return request.body;
  }
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function prepareUpload(body) {
  const { session, filename, bytes } = validateUploadRequest(body);
  const pathname = `ecg-uploads/${session}/${randomUUID()}-${filename}`;
  const now = Date.now();
  const sourceValidUntil = now + SOURCE_URL_LIFETIME_MS;
  const signedToken = await issueSignedToken({
    pathname,
    operations: ["put", "get", "delete"],
    validUntil: sourceValidUntil,
    maximumSizeInBytes: bytes,
  });
  const [putResult, getResult, deleteResult] = await Promise.all([
    presignUrl(signedToken, {
      access: "private",
      operation: "put",
      pathname,
      validUntil: now + UPLOAD_URL_LIFETIME_MS,
      maximumSizeInBytes: bytes,
      addRandomSuffix: false,
      allowOverwrite: false,
      cacheControlMaxAge: 60,
    }),
    presignUrl(signedToken, {
      access: "private",
      operation: "get",
      pathname,
      validUntil: sourceValidUntil,
      useCache: false,
    }),
    presignUrl(signedToken, {
      access: "private",
      operation: "delete",
      pathname,
      validUntil: sourceValidUntil,
    }),
  ]);
  return {
    put_url: putResult.presignedUrl,
    source_file: {
      filename,
      pathname,
      bytes,
      get_url: getResult.presignedUrl,
      delete_url: deleteResult.presignedUrl,
    },
  };
}

async function removeUpload(body) {
  const pathname = validateBlobPath(String(body?.session || ""), body?.pathname);
  await del(pathname);
  return { deleted: true, pathname };
}

export default async function handler(request, response) {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Use POST for private ECG storage." });
  }
  try {
    const body = await readJson(request);
    if (body.action === "prepare_upload") {
      return response.status(200).json(await prepareUpload(body));
    }
    if (body.action === "delete_upload") {
      return response.status(200).json(await removeUpload(body));
    }
    return response.status(400).json({ error: "Unknown private-storage action." });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Private storage failed.";
    return response.status(400).json({ error: message });
  }
}

export const config = { maxDuration: 30 };
