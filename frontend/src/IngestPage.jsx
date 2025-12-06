import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import useGpuDiagnostics from "./hooks/useGpuDiagnostics";
import "./IngestPage.css";

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    try { return await res.json(); } catch {}
  }
  const raw = await res.text();
  return { nonJson: true, raw };
}

function prettyBytes(n) {
  if (n == null || isNaN(n)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0; let val = n;
  while (val >= 1024 && idx < units.length - 1) { val /= 1024; idx++; }
  return `${val.toFixed(val < 10 && idx > 0 ? 1 : 0)} ${units[idx]}`;
}

function shortHash(h) { return (h || "").slice(0, 8); }
function formatDate(v) { try { const d = new Date(v); return isNaN(d) ? "" : d.toLocaleString(); } catch { return ""; } }

function clampPercent(v, fallback = 0) {
  const n = Number(v);
  if (Number.isFinite(n)) {
    return Math.min(100, Math.max(0, n));
  }
  return fallback;
}

function formatProgressDetails(progress) {
  if (!progress) return "";
  const phaseRaw = typeof progress.phase === "string" ? progress.phase : progress.stage;
  const stageRaw = typeof progress.stage === "string" ? progress.stage : "";
  const phase = (phaseRaw || "").replace(/[:_]/g, " ").trim();
  const stage = (stageRaw || "").replace(/[:_]/g, " ").trim();
  const percent = typeof progress.percent === "number" && !Number.isNaN(progress.percent)
    ? `${Math.round(progress.percent)}%`
    : "";
  const parts = [];
  if (phase) parts.push(phase.toUpperCase());
  if (percent) parts.push(percent);
  if (stage && stage.toLowerCase() !== phase.toLowerCase()) {
    parts.push(`· ${stage}`);
  }
  const current = progress.current ?? progress.processed;
  const total = progress.total ?? progress.total_items ?? progress.total_chunks;
  if (typeof current === "number" && typeof total === "number" && total > 0) {
    parts.push(`(${current}/${total})`);
  } else if (typeof progress.chunk_count === "number") {
    parts.push(`(${progress.chunk_count} chunk${progress.chunk_count === 1 ? "" : "s"})`);
  }
  return parts.join(" ").trim();
}

const styles = {
  page: {
    display: "grid",
    gridTemplateColumns: "var(--ingest-grid-columns, minmax(0, 1.65fr) minmax(320px, 1.1875fr))",
    gap: 16,
    alignItems: "stretch",
    width: "100%",
    minHeight: "var(--ingest-left-min-height, calc(100vh - 64px))",
  },
  leftColumn: {
    display: "grid",
    gridTemplateRows: "auto minmax(0, 1fr)",
    gap: 14,
    minWidth: 0,
    minHeight: "var(--ingest-left-min-height, calc(100vh - 64px))",
    height: "var(--ingest-left-min-height, calc(100vh - 64px))",
  },
  uploadCard: {
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    gap: 8,
    padding: "8px 8px",
  },
  uploadProgressList: {
    marginTop: 8,
    flex: 1,
    minHeight: 0,
    maxHeight: "var(--ingest-upload-scroll-height, calc(5 * 58px))",
    overflow: "auto",
    paddingRight: 4,
  },
  previewCard: {
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "hidden",
    padding: "8px 8px",
  },
  previewBody: { flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden", gap: 12 },
  previewContent: { flex: 1, display: "flex", flexDirection: "column", minHeight: 0 },
  previewHeaderRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap", marginBottom: 4 },
  previewTitle: { fontSize: 13, fontWeight: 600, color: "rgba(148, 163, 184, 0.9)" },
  previewLimiterControls: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" },
  previewLimitToggle: { display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "rgba(148, 163, 184, 0.85)" },
  previewLimitCheckbox: { width: 16, height: 16, accentColor: "#818cf8" },
  libraryCard: {
    minWidth: 0,
    minHeight: 0,
    width: "100%",
    display: "flex",
    flexDirection: "column",
    gap: 6,
    padding: "6px 8px 6px 8px",
    position: "var(--ingest-library-position, sticky)",
    top: "var(--ingest-library-top, 16px)",
    height: "var(--ingest-library-height, calc(100vh - 64px))",
    maxHeight: "var(--ingest-library-height, calc(100vh - 64px))",
    minHeight: "var(--ingest-library-height, calc(100vh - 64px))",
    overflow: "hidden",
  },
  card: { border: "none", borderRadius: 26, padding: "24px 26px", background: "linear-gradient(150deg, rgba(78, 89, 162, 0.98), rgba(26, 30, 64, 0.95))", boxShadow: "0 38px 76px rgba(5, 8, 25, 0.78)" },
  sectionHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 12
  },
  sectionTitle: { margin: 0, fontSize: 18, fontWeight: 600, letterSpacing: 0.2, whiteSpace: "nowrap" },
  badge: { padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.4)", fontSize: 12, color: "rgba(148, 163, 184, 0.85)", whiteSpace: "nowrap" },
  row: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" },
  button: { font: "inherit", fontSize: 14, padding: "11px 24px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(167, 139, 250, 0.94), rgba(59, 130, 246, 0.82))", color: "#f8f9ff", cursor: "pointer", boxShadow: "0 24px 44px rgba(5, 9, 25, 0.65)", transition: "transform 0.15s ease, box-shadow 0.15s ease", whiteSpace: "nowrap" },
  filePickerButton: { font: "inherit", fontSize: 14, padding: "11px 24px", borderRadius: 999, border: "none", background: "rgba(23, 28, 60, 0.98)", color: "rgba(248, 250, 255, 0.92)", cursor: "pointer", boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.35), inset 0 2px 14px rgba(3, 6, 18, 0.7)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  hiddenFileInput: { display: "none" },
  subtleButton: { font: "inherit", fontSize: 13, padding: "8px 18px", borderRadius: 999, border: "none", background: "rgba(77, 88, 142, 0.96)", color: "rgba(248, 250, 255, 0.96)", cursor: "pointer", boxShadow: "0 18px 32px rgba(6, 9, 23, 0.62)", transition: "transform 0.15s ease, box-shadow 0.15s ease", whiteSpace: "nowrap" },
  input: { font: "inherit", padding: "12px 20px", borderRadius: 999, border: "none", background: "rgba(23, 28, 60, 0.98)", color: "inherit", minWidth: 0, boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.26), inset 0 2px 14px rgba(3, 6, 18, 0.7)", outline: "none" },
  feedback: { marginTop: 10, fontSize: 13, color: "rgba(148, 163, 184, 0.85)" },
  docs: {
    flex: 1,
    width: "100%",
    minHeight: 0,
    overflowY: "auto",
    overflowX: "hidden",
    border: "none",
    borderRadius: 18,
    padding: 6,
    background: "rgba(21, 26, 58, 0.97)",
    boxShadow: "0 24px 46px rgba(0, 0, 0, 0.6), inset 0 0 0 2px rgba(99, 102, 241, 0.11)",
  },
  listItem: { padding: "10px 16px", borderRadius: 20, background: "rgba(50, 63, 128, 0.96)", border: "none", marginBottom: 10, display: "flex", flexDirection: "column", gap: 4, boxShadow: "0 18px 32px rgba(4, 7, 20, 0.65)" },
  docTitleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 },
  docTitleActions: { display: "flex", alignItems: "center", gap: 8, flexShrink: 0 },
  docName: { fontSize: 15, fontWeight: 600, color: "rgba(226, 232, 240, 0.95)", margin: 0, wordBreak: "break-word", lineHeight: 1.2 },
  docStatusPill: { fontSize: 11, textTransform: "uppercase", letterSpacing: 0.6, padding: "2px 8px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.22)", color: "rgba(148, 163, 184, 0.85)" },
  docMetaRow: { display: "flex", flexWrap: "wrap", gap: 2, fontSize: 12, color: "rgba(148, 163, 184, 0.9)", marginTop: 0, lineHeight: 1.25 },
  docMetaItem: { whiteSpace: "nowrap" },
  docStageRow: { display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 },
  docStageBadge: {
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    padding: "3px 10px",
    borderRadius: 999,
    border: "1px solid rgba(148, 163, 184, 0.32)",
    color: "rgba(226, 232, 240, 0.85)",
  },
  docStageReady: { background: "rgba(34, 197, 94, 0.16)", borderColor: "rgba(34, 197, 94, 0.45)", color: "rgba(209, 250, 229, 0.95)" },
  docStagePending: { background: "rgba(59, 130, 246, 0.12)", borderColor: "rgba(59, 130, 246, 0.45)", color: "rgba(191, 219, 254, 0.95)" },
  docStageError: { background: "rgba(248, 113, 113, 0.14)", borderColor: "rgba(248, 113, 113, 0.55)", color: "rgba(254, 226, 226, 0.92)" },
  docActions: { display: "flex", justifyContent: "flex-end", gap: 1, marginTop: 2 },
  docPreviewButton: { font: "inherit", fontSize: 13, padding: "6px 18px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(165, 180, 252, 0.55), rgba(99, 102, 241, 0.32))", color: "rgba(226, 232, 240, 0.98)", cursor: "pointer", display: "inline-flex", alignItems: "center", justifyContent: "center", boxShadow: "0 16px 28px rgba(3, 6, 19, 0.5)" },
  dangerIconButton: { font: "inherit", fontSize: 16, lineHeight: 1, width: 28, height: 28, borderRadius: 14, border: "none", background: "rgba(252, 165, 165, 0.28)", color: "rgba(255, 241, 242, 0.96)", cursor: "pointer", display: "inline-flex", alignItems: "center", justifyContent: "center", padding: 0, boxShadow: "0 14px 26px rgba(239, 68, 68, 0.38)" },
  docIngestRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, marginTop: 1, fontSize: 12, color: "rgba(148, 163, 184, 0.85)" },
  docIngestInfo: { flex: 1, lineHeight: 1 },
  docPerfToggle: { font: "inherit", fontSize: 12, display: "inline-flex", alignItems: "center", gap: 2, padding: 0, border: "none", background: "transparent", color: "rgba(148, 163, 184, 0.75)", cursor: "pointer" },
  docPerfToggleActive: { color: "rgba(226, 232, 240, 0.95)" },
  docPerfDetails: { marginTop: 1, marginLeft: 12, fontSize: 11, color: "rgba(148, 163, 184, 0.85)", lineHeight: 1.35 },
  muted: { opacity: 0.75, fontSize: 13, color: "rgba(148, 163, 184, 0.8)" },
  error: { fontSize: 13, color: "#ff8f8f", marginTop: 6 },
  progressTrack: { position: "relative", height: 8, background: "rgba(255, 255, 255, 0.12)", borderRadius: 999, overflow: "hidden" },
  progressFill: { position: "absolute", top: 0, bottom: 0, left: 0, background: "linear-gradient(90deg, rgba(167,139,250,0.9), rgba(59,130,246,0.9))", borderRadius: 999, transition: "width 0.4s ease" },
  progressLabel: { marginTop: 6, fontSize: 11, letterSpacing: 0.4, color: "rgba(226, 232, 240, 0.8)", textTransform: "uppercase" },
  miniProgress: { flex: 1, display: "flex", alignItems: "center", gap: 6, minWidth: 140 },
  miniProgressTrack: { flex: 1, height: 6, borderRadius: 999, background: "rgba(255, 255, 255, 0.12)", overflow: "hidden" },
  miniProgressFill: { height: "100%", background: "linear-gradient(90deg, rgba(167,139,250,0.95), rgba(59,130,246,0.85))", transition: "width 0.4s ease" },
  miniProgressLabel: { fontSize: 11, color: "rgba(226, 232, 240, 0.8)", minWidth: 46, textAlign: "right" },
  previewText: {
    border: "none",
    borderRadius: 20,
    background: "rgba(23, 28, 60, 0.97)",
    padding: 12,
    flex: 1,
    minHeight: 0,
    overflow: "auto",
  },
  previewMetaSections: { display: "flex", flexDirection: "column", gap: 6 },
  previewStatsLine: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 2,
    fontSize: 12,
    lineHeight: 1.3,
    color: "rgba(226, 232, 240, 0.9)",
  },
  previewStatsItem: { whiteSpace: "nowrap" },
  previewStatsLabel: { opacity: 0.65, marginRight: 4 },
  previewStatsClassification: { display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap", color: "rgba(226, 232, 240, 0.95)" },
  previewMetaLayout: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 8 },
  previewMetaBlock: {
    borderRadius: 18,
    padding: "10px 12px",
    background: "rgba(25, 33, 74, 0.92)",
    display: "flex",
    flexDirection: "column",
    gap: 6,
    minHeight: 0,
  },
  previewImage: {
    maxWidth: "100%",
    height: "auto",
    display: "block",
    margin: "12px auto",
    borderRadius: 12,
    background: "rgba(15, 23, 42, 0.85)",
    padding: 6,
    border: "1px solid rgba(148, 163, 184, 0.2)",
  },
  previewMetaTitle: { fontSize: 12, letterSpacing: 0.5, textTransform: "uppercase", color: "rgba(148, 163, 184, 0.9)" },
  previewMetaBody: { fontSize: 13, color: "#e2e8f0", lineHeight: 1.45 },
  previewMetaFooter: { fontSize: 12, color: "rgba(148, 163, 184, 0.85)" },
  previewSummaryPlaceholder: { fontStyle: "italic", color: "rgba(148, 163, 184, 0.85)" },
  previewEmpty: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "40px 20px",
    color: "rgba(148, 163, 184, 0.7)",
  },
  markdown: { fontSize: 14, lineHeight: 1.6, color: "#f8fbff", whiteSpace: "normal", wordBreak: "break-word" },
  markdownTable: { width: "100%", borderCollapse: "collapse", margin: "12px 0" },
  tableCell: { border: "1px solid rgba(148, 163, 184, 0.18)", padding: "8px 10px", textAlign: "left" },
  inlineCode: {
    background: "rgba(15, 23, 42, 0.6)",
    borderRadius: 8,
    padding: "2px 6px",
    fontSize: 13,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
  codeBlock: {
    background: "rgba(15, 23, 42, 0.75)",
    borderRadius: 16,
    padding: "14px 16px",
    margin: "12px 0",
    overflowX: "auto",
    fontSize: 13,
    border: "1px solid rgba(148, 163, 184, 0.25)",
  },
};

const markdownRemarkPlugins = [remarkGfm, remarkMath];
const markdownRehypePlugins = [rehypeRaw, rehypeKatex];
const baseMarkdownComponents = {
  table: (props) => <table style={styles.markdownTable} {...props} />,
  th: (props) => <th style={styles.tableCell} {...props} />,
  td: (props) => <td style={styles.tableCell} {...props} />,
  code: ({ inline, children = [], ...props }) =>
    inline ? (
      <code style={styles.inlineCode} {...props}>
        {children}
      </code>
    ) : (
      <pre style={styles.codeBlock}>
        <code {...props}>{String(children).replace(/\n$/, "")}</code>
      </pre>
    ),
};

const FALLBACK_PARSER = "mineru";
const IN_PROGRESS_STATUSES = new Set([
  "processing",
  "ingesting",
  "queued",
  "pending",
  "running",
  "parsing",
  "uploading",
  "ocr_pending",
  "ocr_running",
  "ocr_completed",
  "waiting_postprocess",
  "chunking",
  "embedding",
]);
const COMPLETED_STATUSES = new Set(["processed", "done", "completed", "ready"]);
const ERROR_STATUSES = new Set(["error", "failed"]);

function normalizeStatus(value) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function formatStatusLabel(value) {
  const normalized = normalizeStatus(value);
  if (!normalized) return "";
  return normalized.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().toUpperCase();
}

export default function IngestPage({ systemStatus = {} }) {
  const navigate = useNavigate();
  const api = useMemo(() => ({
    ingest: "/api/ingest",
    docs: "/api/documents",
    status: (jobId) => `/api/status/${jobId}`,
    retry: (hash) => `/api/ingest/${hash}/retry`,
    preprocess: (hash) => `/api/ingest/${hash}/preprocess`,
    reprocessAll: "/api/ingest/reprocess_all",
    classify: (hash) => `/api/ingest/${hash}/classify`,
    previewText: (hash, maxChars = null, parser = FALLBACK_PARSER) => {
      const params = new URLSearchParams({ parser });
      if (typeof maxChars === "number" && Number.isFinite(maxChars)) {
        params.set("max_chars", String(maxChars));
      }
      return `/api/debug/parsed_text/${hash}?${params.toString()}`;
    },
    extractionStats: (schemaId) => `/api/extraction/stats/${schemaId}`,
    extractionExtract: "/api/extraction/extract",
    extractionMetadata: (schemaId, docHash) => `/api/extraction/metadata/${schemaId}/${docHash}`,
  }), []);

  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState([]);
  const [activeJobs, setActiveJobs] = useState(new Set());
  const [retryingHash, setRetryingHash] = useState(null);
  const [reprocessingHash, setReprocessingHash] = useState(null);
  const [reprocessingAll, setReprocessingAll] = useState(false);
  const [classifyingHash, setClassifyingHash] = useState(null);
  const [deletingHash, setDeletingHash] = useState(null);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [preview, setPreview] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [previewInfo, setPreviewInfo] = useState(null);
  const [previewAssets, setPreviewAssets] = useState(null);
  const [previewMaxChars, setPreviewMaxChars] = useState(2000);
  const [limitPreview, setLimitPreview] = useState(true);
  const parser = FALLBACK_PARSER;
  const [expandedPerf, setExpandedPerf] = useState(new Set());
  const [activeDiagnosticsPanel, setActiveDiagnosticsPanel] = useState(null);
  // Extraction state
  const [extractionStats, setExtractionStats] = useState(null);
  const [extractingAll, setExtractingAll] = useState(false);
  const [extractingHash, setExtractingHash] = useState(null);
  const [extractedMetadata, setExtractedMetadata] = useState(null);
  const [extractedDocsSet, setExtractedDocsSet] = useState(new Set());
  const fileInputRef = useRef(null);
  const selectedDocRef = useRef(null);
  const lastPreviewParamsRef = useRef({ previewMaxChars, limitPreview });

  useEffect(() => {
    selectedDocRef.current = selectedDoc;
  }, [selectedDoc]);

  const readyDocCount = Number(systemStatus.docs_count || 0);
  const canOpenChat = systemStatus.ready && !systemStatus.has_running_jobs && readyDocCount > 0;
  const chatDisabledReason = systemStatus.has_running_jobs
    ? "Finish processing the current jobs before chatting."
    : !systemStatus.ready
      ? "System is still preparing the chat experience."
      : readyDocCount === 0
        ? "Ingest at least one document to unlock chat."
        : "";

  const resolveAssetSrc = useCallback((src) => {
    if (src == null) return src;
    const trimmed = String(src).trim();
    if (!trimmed) return trimmed;
    if (/^[a-z]+:/i.test(trimmed) || trimmed.startsWith("data:") || trimmed.startsWith("//")) {
      return trimmed;
    }
    if (trimmed.startsWith("/") && !trimmed.startsWith("./") && !trimmed.startsWith("../")) {
      return trimmed;
    }
    const baseUrl = previewAssets?.base_url;
    if (!baseUrl) return trimmed;
    let normalized = trimmed.replace(/^(\.\/)+/, "");
    normalized = normalized.replace(/^\/+/, "");
    if (!normalized || normalized.startsWith("..")) {
      return trimmed;
    }
    const separator = baseUrl.endsWith("/") ? "" : "/";
    return `${baseUrl}${separator}${normalized}`;
  }, [previewAssets]);

  const markdownComponents = useMemo(() => ({
    ...baseMarkdownComponents,
    img: ({ node, ...props }) => {
      const { src, style, ...rest } = props;
      const resolved = resolveAssetSrc(src);
      return (
        <img
          {...rest}
          src={resolved}
          style={{ ...styles.previewImage, ...(style || {}) }}
          alt={props.alt || ""}
        />
      );
    },
  }), [resolveAssetSrc]);

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try { const res = await fetch(api.docs); const data = await readJsonSafe(res); if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET /docs ${res.status}`); setDocs(Array.isArray(data) ? data : []); }
    catch { setDocs([]); }
    finally { setDocsLoading(false); }
  }, [api.docs]);

  useEffect(() => { void refreshDocs(); }, [refreshDocs]);

  const systemDocs = useMemo(() => (Array.isArray(systemStatus.documents) ? systemStatus.documents : []), [systemStatus.documents]);
  const settingsGroups = useMemo(() => {
    const base = systemStatus.settings || systemStatus.settings_snapshot;
    const merged = base ? { ...base } : {};
    if (systemStatus.gpu_phase) {
      merged.gpu = {
        state: systemStatus.gpu_phase.state || "unknown",
        last_error: systemStatus.gpu_phase.last_error || "",
      };
    }
    return Object.keys(merged).length > 0 ? merged : null;
  }, [systemStatus.settings, systemStatus.settings_snapshot, systemStatus.gpu_phase]);

  const { data: gpuStats, error: gpuError, loading: gpuLoading } = useGpuDiagnostics(activeDiagnosticsPanel === "gpu");
  const displayDocs = docs.length ? docs : systemDocs;
  useEffect(() => {
    if (!selectedDoc?.hash) return;
    const updated = displayDocs.find((doc) => doc && doc.hash === selectedDoc.hash);
    if (updated && updated !== selectedDoc) {
      setSelectedDoc(updated);
    }
  }, [displayDocs, selectedDoc]);
  const jobInfoByHash = useMemo(() => {
    const map = new Map();
    if (Array.isArray(systemStatus.jobs)) {
      systemStatus.jobs.forEach((job) => {
        if (!job || !Array.isArray(job.docs)) return;
        job.docs.forEach((docEntry) => {
          if (docEntry && docEntry.hash) {
            map.set(docEntry.hash, {
              ...docEntry,
              job_id: job.job_id,
              job_status: job.status,
            });
          }
        });
      });
    }
    return map;
  }, [systemStatus.jobs]);

  const jobQueueEntries = useMemo(() => {
    if (!Array.isArray(systemStatus.jobs)) return [];
    const entries = [];
    systemStatus.jobs.forEach((job) => {
      if (!job) return;
      const jobStatus = normalizeStatus(job.status);
      if (!jobStatus || !IN_PROGRESS_STATUSES.has(jobStatus)) return;
      const docsList = Array.isArray(job.docs) ? job.docs : [];
      docsList.forEach((doc, idx) => {
        if (!doc) return;
        const docStatus = normalizeStatus(doc.status);
        if (docStatus && (COMPLETED_STATUSES.has(docStatus) || ERROR_STATUSES.has(docStatus))) {
          return;
        }
        const docHash = doc.hash || doc.doc_hash || doc.docHash;
        const docName = doc.file || doc.name || docHash || `Document ${idx + 1}`;
        entries.push({
          key: docHash || `job:${job.job_id}:${idx}`,
          id: docHash || `job:${job.job_id}:${idx}`,
          name: docName,
          status: doc.status || job.status || "processing",
          jobId: job.job_id,
          jobHash: docHash || null,
          jobProgress: (doc.progress && typeof doc.progress === "object") ? doc.progress : job.progress || null,
          source: "job",
        });
      });
    });
    return entries;
  }, [systemStatus.jobs]);

  const documentQueueEntries = useMemo(() => {
    const docsList = Array.isArray(systemStatus.documents) ? systemStatus.documents : [];
    return docsList
      .filter((doc) => IN_PROGRESS_STATUSES.has(normalizeStatus(doc?.status)))
      .map((doc, idx) => {
        const docHash = doc.hash || doc.doc_hash || doc.docHash;
        const key = docHash || doc.stored_name || doc.name || `doc:${idx}`;
        return {
          key,
          id: key,
          name: doc.name || doc.stored_name || docHash || `Document ${idx + 1}`,
          status: doc.status || "processing",
          jobId: null,
          jobHash: docHash || null,
          jobProgress: { phase: doc.status, stage: doc.status },
          source: "doc",
        };
      });
  }, [systemStatus.documents]);

  const combinedProgressEntries = useMemo(() => {
    const seen = new Set();
    const results = [];

    const addEntry = (entry) => {
      if (!entry) return;
      const key = entry.key || entry.jobHash || entry.id || entry.name;
      if (!key || seen.has(key)) return;
      seen.add(key);
      results.push(entry);
    };

    jobQueueEntries.forEach(addEntry);
    uploadProgress.forEach((entry, idx) => {
      if (!entry) return;
      const key = entry.jobHash || `upload:${entry.jobId || entry.id || idx}`;
      addEntry({ ...entry, key, id: key });
    });
    documentQueueEntries.forEach(addEntry);

    return results;
  }, [jobQueueEntries, uploadProgress, documentQueueEntries]);

  const classificationInfo = selectedDoc?.classification || null;
  const classificationStatusRaw = selectedDoc?.classification_status || selectedDoc?.classificationStatus || "";
  const classificationStatus = normalizeStatus(classificationStatusRaw);
  const classificationReady = Boolean(classificationInfo) && classificationStatus === "classified";
  const classificationInProgress = classificationStatus === "running" || classificationStatus === "queued";
  const classificationErrorMessage = selectedDoc?.classification_error || selectedDoc?.classificationError || "";
  const hasSelectedDoc = Boolean(selectedDoc);
  let classificationPrimaryText = hasSelectedDoc ? "Classification pending." : "";
  let classificationSupplementalText = "";
  if (hasSelectedDoc) {
    if (classificationReady) {
      const targetParts = [classificationInfo.l1_name || classificationInfo.l1_id || "Category"];
      if (classificationInfo.l2_name) targetParts.push(`→ ${classificationInfo.l2_name}`);
      else if (classificationInfo.l2_id && !classificationInfo.l2_name) targetParts.push(`→ ${classificationInfo.l2_id}`);
      classificationPrimaryText = targetParts.join(" ");
    } else if (classificationStatus === "error") {
      classificationPrimaryText = "Classification failed.";
      classificationSupplementalText = classificationErrorMessage || "";
    } else if (classificationInProgress) {
      classificationPrimaryText = "Classification running…";
      classificationSupplementalText = "Hang tight—this will update once the model finishes.";
    } else {
      classificationPrimaryText = "Classification pending.";
      classificationSupplementalText = "Trigger classification from the document library to populate this section.";
    }
  }
  const classificationInlineText = hasSelectedDoc
    ? (classificationSupplementalText
        ? `${classificationPrimaryText} (${classificationSupplementalText})`
        : classificationPrimaryText)
    : "";
  const summaryRaw = typeof selectedDoc?.summary === "string" ? selectedDoc.summary.trim() : "";
  const summaryDisplayText = summaryRaw || "Summary will be shown here.";

  const handleUploadAndIngest = useCallback(
    async (selectedFiles) => {
      const files = Array.isArray(selectedFiles) ? selectedFiles : [];
      if (files.length === 0) {
        setUploadStatus("No files selected.");
        return;
      }

      setUploading(true);
      setUploadStatus(`Uploading ${files.length} file(s)...`);

      // Initialize progress tracking
      const progress = files.map((f, idx) => ({
        id: idx,
        name: f.name,
        status: "pending",
        jobId: null,
        error: null,
        jobProgress: null,
        jobHash: null,
      }));
      setUploadProgress(progress);

      const validFiles = [];
      files.forEach((file, idx) => {
        const isPdf = file && (file.type === "application/pdf" || /\.pdf$/i.test(file.name));
        if (!isPdf) {
          setUploadProgress((prev) =>
            prev.map((p, i) =>
              i === idx ? { ...p, status: "error", error: "Only PDF files are supported" } : p
            )
          );
          return;
        }
        validFiles.push({ file, idx });
        setUploadProgress((prev) =>
          prev.map((p, i) => (i === idx ? { ...p, status: "uploading" } : p))
        );
      });

      if (validFiles.length === 0) {
        setUploading(false);
        setUploadStatus("No valid PDF files to upload.");
        return;
      }

      try {
        const form = new FormData();
        validFiles.forEach(({ file }) => form.append("files", file));
        const res = await fetch(api.ingest, { method: "POST", body: form });
        const data = await readJsonSafe(res);
        if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);

        const results = Array.isArray(data.results) ? [...data.results] : [];
        setUploadProgress((prev) => {
          const queue = [...results];
          return prev.map((item) => {
            if (queue.length === 0) return item;
            if (item.status === "uploading" || item.status === "pending") {
              const result = queue.shift();
              if (!result) return item;
              const status = result.status || item.status;
              const error = result.message || result.error || item.error;
              return {
                ...item,
                status,
                error: error || null,
                jobId: result.job_id || data.job_id || item.jobId,
                jobHash: result.hash || item.jobHash,
                jobProgress: null,
              };
            }
            return item;
          });
        });

        if (data.job_id) {
          setActiveJobs((prev) => new Set([...prev, data.job_id]));
        }

        const queuedCount = results.filter((r) => r && r.status === "queued").length;
        const skippedCount = results.filter((r) => r && r.status === "skipped").length;
        if (queuedCount > 0) {
          setUploadStatus(`Queued ${queuedCount} document${queuedCount === 1 ? "" : "s"}.`);
        } else if (skippedCount > 0) {
          setUploadStatus(`Skipped ${skippedCount} already ingested document${skippedCount === 1 ? "" : "s"}.`);
        } else {
          setUploadStatus("Upload finished.");
        }

        void refreshDocs();
      } catch (err) {
        setUploadStatus(err.message || String(err));
      } finally {
        setUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [api.ingest, refreshDocs]
  );

  const handleFileInputChange = useCallback(
    (event) => {
      const selectedFiles = Array.from(event?.target?.files || []);
      if (event?.target) {
        event.target.value = "";
      }
      if (selectedFiles.length === 0) {
        setUploadStatus("No files selected.");
        return;
      }
      void handleUploadAndIngest(selectedFiles);
    },
    [handleUploadAndIngest]
  );

  const handleRetry = useCallback(async (hash) => {
    if (!hash) return; 
    setRetryingHash(hash); 
    setUploadStatus(`Retrying ingestion for ${shortHash(hash)}...`);
    try { 
      const res = await fetch(api.retry(hash), { method: "POST" }); 
      const data = await readJsonSafe(res); 
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText); 
      if (data.job_id) {
        setActiveJobs(prev => new Set([...prev, data.job_id]));
        setUploadStatus(`Re-queued (job ${data.job_id})`);
      }
      void refreshDocs(); 
    }
    catch (e) { setUploadStatus(`Retry failed: ${e.message || String(e)}`); }
    finally { setRetryingHash(null); }
  }, [api, refreshDocs]);

  const handleRetryPreprocess = useCallback(async (hash) => {
    if (!hash) return;
    setReprocessingHash(hash);
    const short = shortHash(hash);
    setUploadStatus(`Queueing preprocessing for ${short}...`);
    try {
      const res = await fetch(api.preprocess(hash), { method: "POST" });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const jobId = data.job_id;
      const queuedDocs = Number(data.queued_docs ?? data.total_docs ?? 1) || 1;
      if (jobId) {
        setActiveJobs(prev => new Set([...prev, jobId]));
        const summary = [`Queued preprocessing for ${short}`];
        if (queuedDocs > 1) summary.push(`${queuedDocs} docs`);
        summary.push(`job ${jobId}`);
        setUploadStatus(summary.join(" • "));
      } else {
        setUploadStatus(`No preprocessing needed for ${short}.`);
      }
      void refreshDocs();
    } catch (e) {
      setUploadStatus(`Preprocess retry failed: ${e.message || String(e)}`);
    } finally {
      setReprocessingHash(null);
    }
  }, [api.preprocess, refreshDocs]);

  const handleReprocessAll = useCallback(async () => {
    setReprocessingAll(true);
    setUploadStatus("Queueing bulk reprocess + reclassify...");
    try {
      const res = await fetch(api.reprocessAll, { method: "POST" });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const jobId = data.postprocess_job_id || data.job_id;
      const classificationJobId = data.classification_job_id;
      const queuedDocs = Number(data.queued_docs ?? data.total_docs ?? 0) || 0;
      const skipped = Number(data.skipped ?? 0) || 0;
      if (jobId || classificationJobId) {
        setActiveJobs(prev => {
          const next = new Set(prev);
          if (jobId) next.add(jobId);
          if (classificationJobId) next.add(classificationJobId);
          return next;
        });
        const summaryParts = [`Queued reprocess + reclassify for ${queuedDocs} document${queuedDocs === 1 ? "" : "s"}`];
        if (skipped > 0) summaryParts.push(`${skipped} skipped`);
        if (jobId) summaryParts.push(`chunks job ${jobId}`);
        if (classificationJobId && classificationJobId !== jobId) {
          summaryParts.push(`class job ${classificationJobId}`);
        }
        setUploadStatus(summaryParts.join(" • "));
      } else {
        if (skipped > 0) {
          setUploadStatus(`No eligible documents. ${skipped} skipped.`);
        } else {
          setUploadStatus("No documents to reprocess/reclassify.");
        }
      }
      void refreshDocs();
    } catch (e) {
      setUploadStatus(`Bulk reprocess failed: ${e.message || String(e)}`);
    } finally {
      setReprocessingAll(false);
    }
  }, [api.reprocessAll, refreshDocs]);

  const handleReclassify = useCallback(async (hash) => {
    if (!hash) return;
    const short = shortHash(hash);
    setClassifyingHash(hash);
    setUploadStatus(`Reclassifying ${short}...`);
    try {
      const res = await fetch(api.classify(hash), { method: "POST" });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const jobId = data.job_id;
      if (jobId) {
        setActiveJobs(prev => new Set([...prev, jobId]));
        setUploadStatus(`Classification queued for ${short} • job ${jobId}`);
      } else {
        setUploadStatus(`Classification queued for ${short}`);
      }
      await refreshDocs();
    } catch (e) {
      setUploadStatus(`Classification failed: ${e.message || String(e)}`);
    } finally {
      setClassifyingHash(null);
    }
  }, [api.classify, refreshDocs]);

  const handleDeleteDoc = useCallback(async (doc) => {
    if (!doc?.hash) return;
    const hash = doc.hash;
    const displayName = doc.name || doc.stored_name || shortHash(hash);
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(`Remove "${displayName}" and all derived data? This cannot be undone.`);
      if (!confirmed) return;
    }

    setDeletingHash(hash);
    setUploadStatus(`Removing ${displayName}...`);
    try {
      const res = await fetch(`${api.docs}/${hash}`, { method: "DELETE" });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      setUploadStatus(`Removed ${displayName}.`);
      if (selectedDoc && selectedDoc.hash === hash) {
        setSelectedDoc(null);
        setPreview("");
        setPreviewInfo(null);
        setPreviewError("");
        setPreviewAssets(null);
      }
      setExpandedPerf(prev => {
        const next = new Set(prev);
        next.delete(hash);
        return next;
      });
      await refreshDocs();
    } catch (err) {
      setUploadStatus(`Remove failed: ${err.message || String(err)}`);
    } finally {
      setDeletingHash(null);
    }
  }, [api.docs, refreshDocs, selectedDoc]);

  const handlePreview = useCallback(async (doc, { skipSelect = false, preserveContent = false } = {}) => {
    if (!doc || !doc.hash) return;

    if (!skipSelect) {
      setSelectedDoc(prev => (prev && prev.hash === doc.hash ? prev : doc));
    }

    if (!preserveContent) {
      setPreview("");
      setPreviewInfo(null);
      setPreviewAssets(null);
    }
    setPreviewError("");
    setPreviewLoading(true);

    try {
      const maxCharsParam = limitPreview ? previewMaxChars : 0;
      const res = await fetch(api.previewText(doc.hash, maxCharsParam, parser));
      const data = await readJsonSafe(res);
      if (!res.ok) {
        throw new Error((data && (data.detail || data.error || data.raw)) || `GET preview ${res.status}`);
      }
      const totalEmbeddings = Number(data.total_embeddings ?? data.chunk_count ?? data.small_embeddings ?? 0) || 0;
      const hasSmall = typeof data.small_embeddings === "number";
      const smallEmbeddings = hasSmall ? Number(data.small_embeddings) : totalEmbeddings;
      setPreview(typeof data.preview === "string" ? data.preview : "");
      setPreviewInfo({
        document_name: data.document_name,
        file_size: data.file_size,
        extracted_chars: data.extracted_chars,
        total_tokens: typeof data.total_tokens === "number" ? data.total_tokens : 0,
        chunk_count: data.chunk_count,
        total_embeddings: totalEmbeddings,
        small_embeddings: smallEmbeddings,
        preview_chars: data.preview_chars,
        truncated: !!data.truncated,
        parser,
      });
      setPreviewAssets(data.assets && typeof data.assets === "object" ? data.assets : null);
    } catch (e) {
      setPreviewError(e.message || String(e));
      setPreviewAssets(null);
    } finally {
      setPreviewLoading(false);
    }
  }, [api, previewMaxChars, limitPreview, parser]);

  useEffect(() => {
    const prev = lastPreviewParamsRef.current;
    const hasMaxChanged = prev.previewMaxChars !== previewMaxChars;
    const hasLimitChanged = prev.limitPreview !== limitPreview;
    const doc = selectedDocRef.current;

    if (!doc?.hash) {
      lastPreviewParamsRef.current = { previewMaxChars, limitPreview };
      return;
    }
    if (!hasMaxChanged && !hasLimitChanged) {
      return;
    }

    lastPreviewParamsRef.current = { previewMaxChars, limitPreview };
    void handlePreview(doc, { skipSelect: true, preserveContent: true });
  }, [previewMaxChars, limitPreview, handlePreview]);

  // Poll active jobs
  useEffect(() => {
    if (activeJobs.size === 0) return;
    
    const id = setInterval(async () => {
      const jobsToCheck = Array.from(activeJobs);
      const results = await Promise.all(
        jobsToCheck.map(async (jobId) => {
          try {
            const res = await fetch(api.status(jobId));
            const data = await readJsonSafe(res);
            if (res.ok && data.status) {
              const st = String(data.status).toLowerCase();
              return { jobId, status: st, payload: data };
            }
          } catch {}
          return { jobId, status: null, payload: null };
        })
      );
      
      const jobDocsMap = new Map();
      results.forEach((r) => {
        if (r && r.jobId && r.payload) {
          jobDocsMap.set(r.jobId, {
            jobStatus: r.status,
            payload: r.payload,
            docs: Array.isArray(r.payload.docs) ? r.payload.docs : [],
          });
        }
      });
      if (jobDocsMap.size > 0) {
        setUploadProgress(prev =>
          prev.map((entry) => {
            if (!entry.jobId || !jobDocsMap.has(entry.jobId)) return entry;
            const info = jobDocsMap.get(entry.jobId);
            const docMatch = info.docs.find((doc) => (doc && (doc.hash === entry.jobHash || doc.file === entry.name)));
            const nextProgress = docMatch?.progress || entry.jobProgress;
            const nextHash = docMatch?.hash || entry.jobHash;
            const nextStatus = docMatch?.status ? String(docMatch.status).toLowerCase() : entry.status;
            return {
              ...entry,
              status: nextStatus || entry.status,
              jobProgress: nextProgress || entry.jobProgress,
              jobHash: nextHash || entry.jobHash,
            };
          })
        );
      }
      
      // Remove completed jobs
      const completedJobs = results.filter(r => r.status === "done" || r.status?.startsWith("error"));
      if (completedJobs.length > 0) {
        setActiveJobs(prev => {
          const newSet = new Set(prev);
          completedJobs.forEach(j => newSet.delete(j.jobId));
          return newSet;
        });
        void refreshDocs();
      }
    }, 2000);
    
    return () => clearInterval(id);
  }, [activeJobs, api, refreshDocs]);

  const handleFilePickerClick = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

  const handleOpenChat = useCallback(() => {
    if (!canOpenChat) return;
    navigate("/chat");
  }, [canOpenChat, navigate]);

  const handleDiagnosticsPanelChange = useCallback((panelKey) => setActiveDiagnosticsPanel(panelKey), []);

  // Fetch extraction stats
  const refreshExtractionStats = useCallback(async () => {
    try {
      const res = await fetch(api.extractionStats("invoices"));
      if (!res.ok) return;
      const data = await readJsonSafe(res);
      setExtractionStats(data);
      if (Array.isArray(data.extracted_docs)) {
        setExtractedDocsSet(new Set(data.extracted_docs));
      }
    } catch {
      // Extraction stats not available
    }
  }, [api]);

  // Refresh extraction stats on mount and after docs refresh
  useEffect(() => {
    void refreshExtractionStats();
  }, [refreshExtractionStats, docs]);

  // Handle bulk extraction
  const handleExtractAll = useCallback(async () => {
    setExtractingAll(true);
    setUploadStatus("Running metadata extraction...");
    try {
      const res = await fetch(api.extractionExtract, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ schema_id: "invoices" }),
      });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const extracted = Number(data.extracted ?? 0);
      const skipped = Number(data.skipped ?? 0);
      const errors = Number(data.errors ?? 0);
      const parts = [`Extracted ${extracted} document${extracted === 1 ? "" : "s"}`];
      if (skipped > 0) parts.push(`${skipped} skipped`);
      if (errors > 0) parts.push(`${errors} error${errors === 1 ? "" : "s"}`);
      setUploadStatus(parts.join(" • "));
      void refreshExtractionStats();
    } catch (e) {
      setUploadStatus(`Extraction failed: ${e.message || String(e)}`);
    } finally {
      setExtractingAll(false);
    }
  }, [api.extractionExtract, refreshExtractionStats]);

  // Handle single document extraction
  const handleExtractDoc = useCallback(async (hash) => {
    if (!hash) return;
    const short = shortHash(hash);
    setExtractingHash(hash);
    setUploadStatus(`Extracting metadata for ${short}...`);
    try {
      const res = await fetch(api.extractionExtract, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ schema_id: "invoices", doc_hashes: [hash] }),
      });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const extracted = Number(data.extracted ?? 0);
      if (extracted > 0) {
        setUploadStatus(`Extracted metadata for ${short}`);
        setExtractedDocsSet(prev => new Set([...prev, hash]));
      } else {
        setUploadStatus(`No metadata extracted for ${short}`);
      }
      void refreshExtractionStats();
    } catch (e) {
      setUploadStatus(`Extraction failed: ${e.message || String(e)}`);
    } finally {
      setExtractingHash(null);
    }
  }, [api.extractionExtract, refreshExtractionStats]);

  // Fetch extracted metadata for selected document
  const fetchExtractedMetadata = useCallback(async (hash) => {
    if (!hash) {
      setExtractedMetadata(null);
      return;
    }
    try {
      const res = await fetch(api.extractionMetadata("invoices", hash));
      if (!res.ok) {
        setExtractedMetadata(null);
        return;
      }
      const data = await readJsonSafe(res);
      setExtractedMetadata(data);
    } catch {
      setExtractedMetadata(null);
    }
  }, [api]);

  // Fetch metadata when selected doc changes
  useEffect(() => {
    if (selectedDoc?.hash) {
      void fetchExtractedMetadata(selectedDoc.hash);
    } else {
      setExtractedMetadata(null);
    }
  }, [selectedDoc?.hash, fetchExtractedMetadata]);

  // Extraction stats computed values
  const invoiceDocsCount = useMemo(() => {
    // Count docs classified as invoices
    return displayDocs.filter(d => {
      const classInfo = d.classification || {};
      return classInfo.l2_id === "invoices" || classInfo.l2_name?.toLowerCase() === "invoices";
    }).length;
  }, [displayDocs]);

  const pendingExtractionCount = useMemo(() => {
    return displayDocs.filter(d => {
      const classInfo = d.classification || {};
      const isInvoice = classInfo.l2_id === "invoices" || classInfo.l2_name?.toLowerCase() === "invoices";
      return isInvoice && !extractedDocsSet.has(d.hash);
    }).length;
  }, [displayDocs, extractedDocsSet]);

  return (
    <div style={{ position: "relative" }}>
      <DiagnosticsPanel
        activePanel={activeDiagnosticsPanel}
        onToggle={handleDiagnosticsPanelChange}
        groups={settingsGroups}
        gpu={gpuStats}
        gpuError={gpuError}
        gpuLoading={gpuLoading}
      />
      <div className="ingest-layout" style={styles.page}>
      <div style={styles.leftColumn}>
        <section style={{ ...styles.card, ...styles.uploadCard }}>
          <div style={{ ...styles.row, marginBottom: 8, justifyContent: "space-between", alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <input
                ref={fileInputRef}
                type="file" 
                multiple 
                accept=".pdf,application/pdf"
                onChange={handleFileInputChange}
                style={styles.hiddenFileInput}
              />
              <button
                type="button"
                onClick={handleFilePickerClick}
                disabled={uploading}
                style={{ ...styles.button, opacity: uploading ? 0.6 : 1 }}
              >
                {uploading ? "Uploading…" : "Upload & Ingest"}
              </button>
            </div>
            <button
              onClick={handleOpenChat}
              disabled={!canOpenChat}
              style={{ ...styles.button, padding: "11px 24px", opacity: canOpenChat ? 1 : 0.5 }}
              title={canOpenChat ? "Jump to chat" : chatDisabledReason}
            >
              Open Chat
            </button>
          </div>
          {combinedProgressEntries.length > 0 && (
            <div style={styles.uploadProgressList}>
              {combinedProgressEntries.map((p) => {
                const uploadWidth = clampPercent(p.jobProgress?.percent, 0);
                const uploadLabel = formatProgressDetails(p.jobProgress) || p.status;
                const normalizedStatus = typeof p.status === "string" ? p.status.toLowerCase() : "";
                const statusIcon = (() => {
                  if (normalizedStatus === "pending") return "⏸";
                  if (normalizedStatus === "uploading") return "⏳";
                  if (normalizedStatus === "queued" || normalizedStatus === "completed" || normalizedStatus === "done") return "✓";
                  if (normalizedStatus === "skipped") return "⊘";
                  if (normalizedStatus.startsWith("error")) return "✗";
                  return "⚙︎";
                })();
                return (
                  <div 
                    key={p.key || p.id || p.jobHash || p.name} 
                    style={{ 
                    fontSize: 12, 
                    padding: "8px 14px", 
                    marginBottom: 4, 
                    background: "rgba(57, 69, 130, 0.95)", 
                    borderRadius: 16,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    boxShadow: "0 18px 30px rgba(4, 7, 20, 0.6)"
                  }}
                  >
                  <span>
                    {statusIcon}
                  </span>
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {p.name}
                  </span>
                  <div style={styles.miniProgress}>
                    <div style={styles.miniProgressTrack}>
                      <div
                        style={{
                          ...styles.miniProgressFill,
                          width: `${uploadWidth}%`,
                        }}
                      />
                    </div>
                    <span style={styles.miniProgressLabel}>
                      {uploadLabel}
                    </span>
                  </div>
                  {p.error && (
                    <span style={{ fontSize: 11, color: "#ff8f8f" }} title={p.error}>
                      {p.error.length > 20 ? p.error.substring(0, 20) + "..." : p.error}
                    </span>
                  )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section style={{ ...styles.card, ...styles.previewCard }}>
        {!selectedDoc && (
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Document Details & Preview</h3>
          </div>
        )}
        <div style={styles.previewBody}>
          {selectedDoc ? (
            <>
              {/* Document Metadata Section */}
              <div style={{ background: "rgba(33, 42, 88, 0.94)", border: "none", borderRadius: 20, padding: 12, marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 18 }}>📄</span>
                  <strong style={{ fontSize: 15, color: "#e2e8f0", flex: 1, minWidth: 0, wordBreak: "break-word" }}>
                    {previewInfo?.document_name || selectedDoc.name}
                  </strong>
                  <span style={styles.badge}>{parser}</span>
                </div>
                
                <div style={styles.previewMetaSections}>
                  {previewInfo && (
                    <div style={styles.previewStatsLine}>
                      <span style={styles.previewStatsItem}>
                        <span style={styles.previewStatsLabel}>Size:</span>
                        {prettyBytes(previewInfo.file_size)}
                      </span>
                      <span style={styles.previewStatsItem}>
                        <span style={styles.previewStatsLabel}>Characters:</span>
                        {(previewInfo.extracted_chars || 0).toLocaleString()}
                      </span>
                      <span style={styles.previewStatsItem}>
                        <span style={styles.previewStatsLabel}>Tokens:</span>
                        {Number(previewInfo.total_tokens || 0).toLocaleString()}
                      </span>
                      <span style={styles.previewStatsItem}>
                        <span style={styles.previewStatsLabel}>Embeddings:</span>
                        {Number(previewInfo.total_embeddings || previewInfo.chunk_count || 0).toLocaleString()}
                      </span>
                      {hasSelectedDoc && (
                        <span style={{ ...styles.previewStatsItem, display: "inline-flex", alignItems: "center", gap: 6 }}>
                          <span style={styles.previewStatsLabel}>Classification:</span>
                          <span style={styles.previewStatsClassification}>
                            {classificationInlineText}
                          </span>
                        </span>
                      )}
                    </div>
                  )}
                  <div style={styles.previewMetaLayout}>
                    <div style={styles.previewMetaBlock}>
                      <span style={styles.previewMetaTitle}>Summary</span>
                      <div
                        style={{
                          ...styles.previewMetaBody,
                          ...(summaryRaw ? {} : styles.previewSummaryPlaceholder),
                        }}
                      >
                        {summaryDisplayText}
                      </div>
                      {!summaryRaw && (
                        <div style={styles.previewMetaFooter}>
                          Summaries will appear once the summarization step is enabled for this document.
                        </div>
                      )}
                    </div>
                    {extractedMetadata && (
                      <div style={{ ...styles.previewMetaBlock, background: "rgba(251, 191, 36, 0.08)" }}>
                        <span style={{ ...styles.previewMetaTitle, color: "rgba(251, 191, 36, 0.9)" }}>
                          📋 Extracted Invoice Data
                        </span>
                        <div style={{ ...styles.previewMetaBody, display: "flex", flexDirection: "column", gap: 4 }}>
                          {extractedMetadata.vendor_name && (
                            <div><span style={{ opacity: 0.65 }}>Vendor:</span> {extractedMetadata.vendor_name}</div>
                          )}
                          {extractedMetadata.invoice_number && (
                            <div><span style={{ opacity: 0.65 }}>Invoice #:</span> {extractedMetadata.invoice_number}</div>
                          )}
                          {extractedMetadata.invoice_date && (
                            <div><span style={{ opacity: 0.65 }}>Date:</span> {extractedMetadata.invoice_date}</div>
                          )}
                          {extractedMetadata.due_date && (
                            <div><span style={{ opacity: 0.65 }}>Due:</span> {extractedMetadata.due_date}</div>
                          )}
                          {(extractedMetadata.total_amount || extractedMetadata.currency) && (
                            <div>
                              <span style={{ opacity: 0.65 }}>Total:</span>{" "}
                              {extractedMetadata.currency && `${extractedMetadata.currency} `}
                              {extractedMetadata.total_amount || "—"}
                            </div>
                          )}
                          {extractedMetadata.line_items_json && (
                            <details style={{ marginTop: 4 }}>
                              <summary style={{ cursor: "pointer", fontSize: 12, opacity: 0.75 }}>
                                Line items ({(() => { try { return JSON.parse(extractedMetadata.line_items_json).length; } catch { return 0; } })()})
                              </summary>
                              <pre style={{ fontSize: 11, margin: "6px 0 0", padding: 8, background: "rgba(0,0,0,0.2)", borderRadius: 8, overflow: "auto", maxHeight: 150 }}>
                                {(() => { try { return JSON.stringify(JSON.parse(extractedMetadata.line_items_json), null, 2); } catch { return extractedMetadata.line_items_json; } })()}
                              </pre>
                            </details>
                          )}
                        </div>
                        {extractedMetadata.extracted_at && (
                          <div style={styles.previewMetaFooter}>
                            Extracted {formatDate(extractedMetadata.extracted_at)}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Text Preview Section */}
              <div style={styles.previewContent}>
                <div style={styles.previewHeaderRow}>
                  <div style={styles.previewTitle}>Extracted Text</div>
                  <div style={styles.previewLimiterControls}>
                    <label style={styles.previewLimitToggle}>
                      <input
                        type="checkbox"
                        checked={limitPreview}
                        onChange={(e) => setLimitPreview(e.target.checked)}
                        style={styles.previewLimitCheckbox}
                      />
                      <span>Limit preview</span>
                    </label>
                    <input
                      type="number"
                      min={200}
                      max={20000}
                      step={100}
                      value={previewMaxChars}
                      onChange={(e) => setPreviewMaxChars(Number(e.target.value) || 2000)}
                      disabled={!limitPreview}
                      style={{ ...styles.input, width: 100, padding: "6px 10px", fontSize: 13, opacity: limitPreview ? 1 : 0.45 }}
                    />
                  </div>
                </div>
                <div style={{ ...styles.previewText, wordBreak: "break-word", overflowWrap: "anywhere" }}>
                  {previewLoading ? (
                    <span style={styles.muted}>Loading…</span>
                  ) : (preview || "").trim().length > 0 ? (
                    <div style={styles.markdown}>
                      <ReactMarkdown
                        remarkPlugins={markdownRemarkPlugins}
                        rehypePlugins={markdownRehypePlugins}
                        components={markdownComponents}
                      >
                        {preview}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <span style={styles.muted}>No text extracted.</span>
                  )}
                </div>
                {previewError && (<div style={styles.error}>Error: {previewError}</div>)}
              </div>
            </>
          ) : (
            <div style={styles.previewEmpty}>
              <div>
                <div style={{ fontSize: 32, marginBottom: 12 }}>📄</div>
                <div>Select a document from the library below to view its details and preview the extracted text.</div>
              </div>
            </div>
          )}
        </div>
        </section>
      </div>

      <section style={{ ...styles.card, ...styles.libraryCard }}>
        <div style={styles.sectionHeader}>
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <h3 style={styles.sectionTitle}>Document Library</h3>
              {!canOpenChat && (
                <span style={{ ...styles.muted, fontSize: 11 }}>{chatDisabledReason}</span>
              )}
            </div>
            <span style={styles.muted}>{`${systemStatus.docs_count || 0} ready / ${systemStatus.total_docs || displayDocs.length}`}</span>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "nowrap", justifyContent: "flex-end", alignItems: "center", minWidth: 0, flexShrink: 0 }}>
            {invoiceDocsCount > 0 && (
              <button
                onClick={handleExtractAll}
                disabled={extractingAll || pendingExtractionCount === 0}
                style={{
                  ...styles.subtleButton,
                  background: "linear-gradient(135deg, rgba(251, 191, 36, 0.85), rgba(245, 158, 11, 0.75))",
                  opacity: extractingAll || pendingExtractionCount === 0 ? 0.5 : 1,
                }}
                title={pendingExtractionCount === 0 ? "All invoices already extracted" : `Extract metadata from ${pendingExtractionCount} invoice${pendingExtractionCount === 1 ? "" : "s"}`}
              >
                {extractingAll ? "Extracting…" : `Extract Invoices${pendingExtractionCount > 0 ? ` (${pendingExtractionCount})` : ""}`}
              </button>
            )}
            <button
              onClick={handleReprocessAll}
              disabled={reprocessingAll || displayDocs.length === 0}
              style={{ ...styles.button, padding: "8px 18px", opacity: reprocessingAll || displayDocs.length === 0 ? 0.5 : 1 }}
              title={displayDocs.length === 0 ? "No documents available" : "Re-chunk and re-embed every document"}
            >
              {reprocessingAll ? "Reprocessing…" : "Reprocess All"}
            </button>
            <button
              onClick={refreshDocs}
              disabled={docsLoading}
              style={{ ...styles.subtleButton, opacity: docsLoading ? 0.6 : 1 }}
            >
              {docsLoading ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </div>
        <div style={{ ...styles.docs, ...(displayDocs.length ? {} : styles.muted) }}>
          {displayDocs.length === 0 ? (
            <div>No documents yet. Upload files to build your knowledge base.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>
              {displayDocs.map((d) => {
                const perf = d.performance;
                const hasPerf = perf && perf.total_time_sec != null;
                const isExpanded = expandedPerf.has(d.hash);
                const statusLabel = normalizeStatus(d.status || "pending") || "pending";
                const jobInfo = d.hash ? jobInfoByHash.get(d.hash) : null;
                const jobStatusLabel = normalizeStatus(jobInfo?.status || jobInfo?.job_status || "");
                const jobProgress = jobInfo?.progress;
                const isErrored = ERROR_STATUSES.has(statusLabel) || ERROR_STATUSES.has(jobStatusLabel) || Boolean(d.error);
                const isInProgress = !isErrored && (
                  IN_PROGRESS_STATUSES.has(statusLabel) ||
                  IN_PROGRESS_STATUSES.has(jobStatusLabel)
                );
                const isCompleted = COMPLETED_STATUSES.has(statusLabel) || COMPLETED_STATUSES.has(jobStatusLabel);
                const canPreviewInHeader = Boolean(d.hash && !isErrored && !isInProgress);
                const showStatusPill = !canPreviewInHeader;
                const showRetry = isErrored && Boolean(d.hash);
                const showPerf = hasPerf && isCompleted;
                const isDeleting = deletingHash === d.hash;
                const showDocProgress = isInProgress && jobProgress && typeof jobProgress.percent === "number";
                const docProgressPercent = clampPercent(jobProgress?.percent, 0);
                const docProgressWidth = docProgressPercent;
                const pillLabel = jobStatusLabel || statusLabel;
                const docProgressLabel = formatProgressDetails(jobProgress) || formatStatusLabel(pillLabel) || formatStatusLabel(statusLabel);
                const ocrAvailable = Boolean(d.ocr_available);
                const embeddingsAvailable = Boolean(d.embedding_available);
                const totalEmbeddings = Number(d.total_embeddings ?? 0) || 0;
                const smallEmbeddings = Number(d.small_embeddings ?? totalEmbeddings) || 0;
                const ocrBadgeLabel = ocrAvailable ? "OCR READY" : isErrored ? "OCR ERROR" : "OCR PENDING";
                const embedBadgeLabel = embeddingsAvailable
                  ? `EMBED READY${totalEmbeddings ? ` (${totalEmbeddings})` : ""}`
                  : isErrored && ocrAvailable
                    ? "EMBED ERROR"
                    : ocrAvailable
                      ? "EMBED PENDING"
                      : "EMBED WAITING";
                const ocrBadgeStyle = ocrAvailable
                  ? styles.docStageReady
                  : isErrored
                    ? styles.docStageError
                    : styles.docStagePending;
                const embedBadgeStyle = embeddingsAvailable
                  ? styles.docStageReady
                  : isErrored && ocrAvailable
                    ? styles.docStageError
                    : styles.docStagePending;
                const showRetryOcr = showRetry && !ocrAvailable;
                const showRetryPreprocess = showRetry && ocrAvailable && !embeddingsAvailable;
                const showGeneralRetry = showRetry && !showRetryOcr && !showRetryPreprocess;
                const classificationStatusRaw = d.classification_status || d.classificationStatus;
                const classificationStatus = normalizeStatus(classificationStatusRaw);
                const classificationInfo = d.classification || null;
                const classificationError = d.classification_error || d.classificationError || "";
                const classificationReady = classificationStatus === "classified" && classificationInfo;
                const classificationInProgress = classificationStatus === "running" || classificationStatus === "queued";
                const classificationLabel = classificationReady
                  ? "CLASSIFICATION READY"
                  : classificationStatus === "error"
                    ? "CLASSIFICATION ERROR"
                    : classificationInProgress
                      ? "CLASSIFYING"
                      : "CLASSIFICATION PENDING";
                const classificationBadgeStyle = classificationStatus === "classified"
                  ? styles.docStageReady
                  : classificationStatus === "error"
                    ? styles.docStageError
                    : styles.docStagePending;
                const classificationTooltip = classificationReady
                  ? [
                      classificationInfo.l1_name || classificationInfo.l1_id,
                      classificationInfo.l2_name ? `→ ${classificationInfo.l2_name}` : "",
                      classificationInfo.l2_id && !classificationInfo.l2_name ? `→ ${classificationInfo.l2_id}` : "",
                      classificationInfo.l1_confidence ? `L1 confidence: ${classificationInfo.l1_confidence}` : "",
                      classificationInfo.l2_confidence ? `L2 confidence: ${classificationInfo.l2_confidence}` : "",
                    ].filter(Boolean).join("\n") || undefined
                  : (classificationError || undefined);
                const canReclassify = Boolean(d.hash && isCompleted);
                const reclassifyDisabled = classificationInProgress || classifyingHash === d.hash;
                const classificationReclassHint = "Click to re-run classification for this document";
                const classificationBadgeTitle = canReclassify
                  ? (classificationTooltip ? `${classificationTooltip}\n\n${classificationReclassHint}` : classificationReclassHint)
                  : classificationTooltip || undefined;
                const classificationBadgeCommonStyle = { ...styles.docStageBadge, ...classificationBadgeStyle };
                
                // Extraction badge for invoice documents
                const isInvoiceDoc = classificationInfo && (classificationInfo.l2_id === "invoices" || classificationInfo.l2_name?.toLowerCase() === "invoices");
                const docIsExtracted = extractedDocsSet.has(d.hash);
                const showExtractionBadge = isInvoiceDoc && isCompleted;
                const extractionBadgeLabel = docIsExtracted ? "EXTRACTED" : "EXTRACTION PENDING";
                const extractionBadgeStyle = docIsExtracted
                  ? { background: "rgba(251, 191, 36, 0.18)", borderColor: "rgba(251, 191, 36, 0.5)", color: "rgba(254, 243, 199, 0.95)" }
                  : styles.docStagePending;
                const canExtractDoc = showExtractionBadge && !docIsExtracted;
                const extractDisabled = extractingHash === d.hash;
                const extractionBadgeTitle = docIsExtracted
                  ? "Metadata has been extracted from this invoice"
                  : "Click to extract metadata from this invoice";
                
                return (
                  <li key={d.hash || d.stored_name || d.name} style={styles.listItem}>
                    <div style={styles.docTitleRow} title={d.path}>
                      <div style={{ flex: 1 }}>
                        <div style={styles.docName}>{d.name || d.stored_name || "Untitled document"}</div>
                      </div>
                      <div style={styles.docTitleActions}>
                        {canPreviewInHeader ? (
                          <button
                            style={{ ...styles.docPreviewButton, opacity: previewLoading && selectedDoc && selectedDoc.hash === d.hash ? 0.6 : 1 }}
                            onClick={() => handlePreview(d)}
                            disabled={previewLoading && selectedDoc && selectedDoc.hash === d.hash}
                            title="Preview extracted text"
                          >
                            {previewLoading && selectedDoc && selectedDoc.hash === d.hash ? "Loading…" : "Preview"}
                          </button>
                        ) : showStatusPill ? (
                          <span style={styles.docStatusPill}>{formatStatusLabel(pillLabel)}</span>
                        ) : null}
                        <button
                          type="button"
                          style={{ ...styles.dangerIconButton, opacity: isDeleting ? 0.5 : 1 }}
                          onClick={() => handleDeleteDoc(d)}
                          disabled={isDeleting}
                          title="Remove document"
                          aria-label={`Remove ${d.name || d.stored_name || "document"}`}
                        >
                          ×
                        </button>
                      </div>
                    </div>

                    <div style={styles.docMetaRow}>
                      <span style={styles.docMetaItem}>
                        <span style={{ opacity: 0.65 }}>Size:</span> {prettyBytes(d.size)}
                      </span>
                      {d.hash && (
                        <span style={styles.docMetaItem}>
                          <span style={{ opacity: 0.65 }}>Hash:</span> {shortHash(d.hash)}
                        </span>
                      )}
                    </div>
                    <div style={styles.docStageRow}>
                      <span
                        style={{ ...styles.docStageBadge, ...ocrBadgeStyle }}
                        title={d.ocr_extracted_at ? `Extracted ${formatDate(d.ocr_extracted_at)}` : undefined}
                      >
                        {ocrBadgeLabel}
                      </span>
                      <span
                        style={{ ...styles.docStageBadge, ...embedBadgeStyle }}
                        title={
                          embeddingsAvailable
                            ? `Chunks: ${smallEmbeddings.toLocaleString()}`
                            : undefined
                        }
                      >
                        {embedBadgeLabel}
                      </span>
                      {canReclassify ? (
                        <button
                          type="button"
                          style={{
                            ...classificationBadgeCommonStyle,
                            cursor: reclassifyDisabled ? "default" : "pointer",
                            opacity: reclassifyDisabled ? 0.6 : 1,
                          }}
                          onClick={() => handleReclassify(d.hash)}
                          disabled={reclassifyDisabled}
                          title={classificationBadgeTitle}
                          aria-label="Re-run document classification"
                        >
                          {classificationLabel}
                        </button>
                      ) : (
                        <span
                          style={classificationBadgeCommonStyle}
                          title={classificationBadgeTitle}
                        >
                          {classificationLabel}
                        </span>
                      )}
                      {showExtractionBadge && (
                        canExtractDoc ? (
                          <button
                            type="button"
                            style={{
                              ...styles.docStageBadge,
                              ...extractionBadgeStyle,
                              cursor: extractDisabled ? "default" : "pointer",
                              opacity: extractDisabled ? 0.6 : 1,
                            }}
                            onClick={() => handleExtractDoc(d.hash)}
                            disabled={extractDisabled}
                            title={extractionBadgeTitle}
                            aria-label="Extract metadata from invoice"
                          >
                            {extractDisabled ? "EXTRACTING…" : extractionBadgeLabel}
                          </button>
                        ) : (
                          <span
                            style={{ ...styles.docStageBadge, ...extractionBadgeStyle }}
                            title={extractionBadgeTitle}
                          >
                            {extractionBadgeLabel}
                          </span>
                        )
                      )}
                    </div>
                    {classificationReady && (
                      <div style={{ ...styles.docMetaRow, marginTop: 4 }}>
                        <span style={styles.docMetaItem}>
                          <span style={{ opacity: 0.65 }}>Classification:</span>{" "}
                          {classificationInfo.l1_name || classificationInfo.l1_id}
                          {classificationInfo.l2_name ? ` → ${classificationInfo.l2_name}` : ""}
                          {classificationInfo.l2_id && !classificationInfo.l2_name ? ` → ${classificationInfo.l2_id}` : ""}
                        </span>
                      </div>
                    )}

                    {(d.last_ingested_at || showPerf) && (
                      <>
                        <div style={styles.docIngestRow}>
                          <span style={styles.docIngestInfo}>
                            {d.last_ingested_at ? `Ingested ${formatDate(d.last_ingested_at)}` : "Ingestion details pending"}
                            {showPerf && (
                              <>
                                {" • "}
                                <span style={{ fontWeight: 600 }}>{perf.total_time_sec.toFixed(1)}s</span>
                              </>
                            )}
                          </span>
                          {showPerf && (
                            <button
                              type="button"
                              onClick={() => {
                                const newSet = new Set(expandedPerf);
                                if (isExpanded) newSet.delete(d.hash);
                                else newSet.add(d.hash);
                                setExpandedPerf(newSet);
                              }}
                              style={{
                                ...styles.docPerfToggle,
                                ...(isExpanded ? styles.docPerfToggleActive : {}),
                              }}
                              title={isExpanded ? "Hide breakdown" : "Show breakdown"}
                              aria-label={`${isExpanded ? "Hide" : "Show"} ingestion breakdown`}
                            >
                              {isExpanded ? "▲" : "▼"}
                            </button>
                          )}
                        </div>

                        {showPerf && isExpanded && (
                          <div style={styles.docPerfDetails}>
                            {perf.ocr_time_sec != null && (
                              <div>• OCR: {perf.ocr_time_sec.toFixed(2)}s</div>
                            )}
                            {perf.ocr_time_sec == null && (
                              <div style={{ opacity: 0.6 }}>• OCR: unavailable</div>
                            )}
                            {perf.chunking_time_sec != null && (
                              <div>• Chunking: {perf.chunking_time_sec.toFixed(2)}s</div>
                            )}
                            {perf.embedding_time_sec != null && (
                              <div>• Embeddings: {perf.embedding_time_sec.toFixed(2)}s</div>
                            )}
                          </div>
                        )}
                      </>
                    )}
                    {showDocProgress && (
                      <div style={{ marginTop: 8 }}>
                        <div style={styles.progressTrack} aria-label="Document progress">
                          <div
                            style={{
                              ...styles.progressFill,
                              width: `${docProgressWidth}%`,
                            }}
                          />
                        </div>
                        <div style={styles.progressLabel}>
                          {docProgressLabel}
                        </div>
                      </div>
                    )}
                    {d.error && (<div style={styles.error}>Error: {d.error}</div>)}
                    {classificationError && classificationStatus === "error" && (
                      <div style={styles.error}>Classification error: {classificationError}</div>
                    )}

                    {showRetry && (
                      <div style={styles.docActions}>
                        {showRetryOcr && (
                          <button
                            style={styles.subtleButton}
                            onClick={() => handleRetry(d.hash)}
                            disabled={retryingHash === d.hash}
                            title="Re-run OCR and ingestion from the source file"
                          >
                            {retryingHash === d.hash ? "Retrying OCR…" : "Retry OCR"}
                          </button>
                        )}
                        {showRetryPreprocess && (
                          <button
                            style={styles.subtleButton}
                            onClick={() => handleRetryPreprocess(d.hash)}
                            disabled={reprocessingHash === d.hash}
                            title="Re-run chunking and embeddings using saved OCR text"
                          >
                            {reprocessingHash === d.hash ? "Retrying…" : "Retry Preprocess"}
                          </button>
                        )}
                        {showGeneralRetry && (
                          <button
                            style={styles.subtleButton}
                            onClick={() => handleRetry(d.hash)}
                            disabled={retryingHash === d.hash}
                          >
                            {retryingHash === d.hash ? "Retrying…" : "Retry"}
                          </button>
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>
    </div>
  </div>
  );
}
