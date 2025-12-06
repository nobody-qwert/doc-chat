import React, { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import RetrievalPanel from "./components/RetrievalPanel";
import useGpuDiagnostics from "./hooks/useGpuDiagnostics";

const ENV_CONTEXT_LIMIT = Number(import.meta.env.VITE_LLM_CONTEXT_SIZE || "10000") || 10000;
const makeDefaultContextStats = () => ({ used: 0, limit: ENV_CONTEXT_LIMIT, truncated: false, ratio: 0 });

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) { try { return await res.json(); } catch {} }
  const raw = await res.text();
  return { nonJson: true, raw };
}

const createMessageId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
function mergeSources(existing = [], incoming = []) {
  const base = Array.isArray(existing) ? existing : [];
  const extra = Array.isArray(incoming) ? incoming : [];
  const combined = [...base, ...extra];
  const deduped = new Map();
  combined.forEach((item, idx) => {
    if (!item) return;
    const key = item.chunk_id || `${item.doc_hash || "na"}-${item.order_index ?? idx}`;
    if (!deduped.has(key)) deduped.set(key, item);
  });
  return Array.from(deduped.values());
}
function formatTiming(t) {
  if (!t) return "";
  const parts = [];
  if (typeof t.timeToFirst === "number") parts.push(`TTFT ${t.timeToFirst.toFixed(2)}s`);
  if (typeof t.generationSeconds === "number") parts.push(`Gen ${t.generationSeconds.toFixed(2)}s`);
  if (typeof t.tokensPerSecond === "number") parts.push(`${t.tokensPerSecond.toFixed(1)} tok/s`);
  return parts.join(" · ");
}
function formatSteps(steps = []) {
  if (!Array.isArray(steps) || steps.length === 0) return "";
  return steps
    .slice()
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    .map((s) => {
      const name = s.name || s.kind || "step";
      const dur = typeof s.duration_seconds === "number" ? `${s.duration_seconds.toFixed(2)}s` : "";
      const ttft = typeof s.time_to_first_token_seconds === "number" ? `TTFT ${s.time_to_first_token_seconds.toFixed(2)}s` : "";
      const tps = typeof s.tokens_per_second === "number" ? `${s.tokens_per_second.toFixed(1)} tok/s` : "";
      const extras = [ttft, tps].filter(Boolean).join(", ");
      const suffix = [dur, extras].filter(Boolean).join(" · ");
      return suffix ? `${name} (${suffix})` : name;
    })
    .filter(Boolean);
}

// Lightweight heuristics that let us guess whether a step detail feels like an input or an output
const STEP_INPUT_HINTS = ["arg", "input", "query", "target", "filter", "prompt"];
const STEP_OUTPUT_HINTS = ["result", "count", "status", "evidence", "decomposition", "answer", "reason", "clarification", "strategy", "duration"];

function humanizeDetailKey(key = "") {
  if (!key) return "";
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function categorizeStepDetails(details) {
  if (!details || typeof details !== "object") {
    return { inputs: [], outputs: [] };
  }
  const inputs = [];
  const outputs = [];
  Object.entries(details).forEach(([key, value]) => {
    if (value == null) return;
    if (key === "args" && value && typeof value === "object") {
      Object.entries(value).forEach(([subKey, subValue]) => {
        if (subValue == null) return;
        inputs.push([humanizeDetailKey(subKey), subValue]);
      });
      return;
    }
    const lower = key.toLowerCase();
    const isInput = STEP_INPUT_HINTS.some((hint) => lower.includes(hint));
    const isOutput = STEP_OUTPUT_HINTS.some((hint) => lower.includes(hint));
    const bucket = isInput && !isOutput ? inputs : outputs;
    bucket.push([humanizeDetailKey(key), value]);
  });
  return { inputs, outputs };
}

function formatStepDetailValue(value) {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

const styles = {
  page: {
    display: "flex",
    justifyContent: "center",
    width: "100%",
    height: "calc(100vh - 32px)",
    maxHeight: "calc(100vh - 32px)",
    overflow: "hidden",
  },
  chatCard: {
    flex: "0 1 75%",
    width: "75%",
    border: "none",
    borderRadius: 24,
    padding: "10px 10px",
    background: "linear-gradient(145deg, rgba(63, 76, 149, 0.98), rgba(22, 26, 55, 0.95))",
    boxShadow: "0 36px 72px rgba(5, 8, 25, 0.78)",
    display: "flex",
    flexDirection: "column",
    gap: 16,
    minHeight: 0,
    maxHeight: "100%",
    overflow: "hidden",
  },
  sectionHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" },
  sectionTitle: { margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 0.25, color: "#ffffff" },
  button: { font: "inherit", fontSize: 14, padding: "10px 22px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.92), rgba(59, 130, 246, 0.78))", color: "#ffffff", cursor: "pointer", boxShadow: "0 20px 40px rgba(8, 12, 32, 0.65)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  subtleButton: { font: "inherit", fontSize: 13, padding: "8px 18px", borderRadius: 999, border: "none", background: "rgba(66, 77, 124, 0.96)", color: "#fdfdff", cursor: "pointer", boxShadow: "0 16px 30px rgba(6, 9, 25, 0.65)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  input: { font: "inherit", padding: "12px 22px", borderRadius: 999, border: "none", background: "rgba(21, 26, 54, 0.98)", color: "#ffffff", flex: 1, minWidth: 0, boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.3), inset 0 2px 14px rgba(3, 6, 18, 0.7)", outline: "none" },
  muted: { fontSize: 13, color: "#f8fbff" },
  messages: { flex: 1, minHeight: 240, minWidth: 0, overflow: "auto", border: "none", borderRadius: 24, padding: 8, background: "rgba(22, 27, 58, 0.98)", whiteSpace: "pre-wrap", display: "flex", flexDirection: "column", gap: 12, boxShadow: "0 22px 48px rgba(3, 5, 15, 0.75), inset 0 0 0 2px rgba(99, 102, 241, 0.14)", maxHeight: "100%" },
  messageList: { display: "flex", flexDirection: "column", gap: 2 },
  userBubble: { alignSelf: "flex-end", background: "rgba(25, 77, 151, 0.95)", borderRadius: 22, padding: 10.5, maxWidth: "85%", boxShadow: "0 20px 40px rgba(3, 8, 23, 0.7)", color: "#fbfcff" },
  assistantBubble: {
    alignSelf: "flex-start",
    background: "rgba(30, 101, 201, 0.9)",
    borderRadius: 22,
    padding: 10.5,
    maxWidth: "95%",
    lineHeight: 1.155,
    boxShadow: "0 20px 40px rgba(3, 8, 23, 0.65)",
    color: "#fbfcff",
  },
  systemBubble: { alignSelf: "center", background: "rgba(99, 102, 241, 0.18)", borderRadius: 18, padding: 10, maxWidth: "85%", lineHeight: 1.35, boxShadow: "0 10px 18px rgba(79, 70, 229, 0.25)", color: "#e0e7ff", border: "1px solid rgba(99, 102, 241, 0.35)" },
  pipelineBubble: {
    alignSelf: "stretch",
    width: "100%",
    background: "transparent",
    padding: "0",
    lineHeight: 1.1,
    color: "#cbd5f5",
    border: "none",
    display: "flex",
    flexDirection: "column",
    gap: 0,
    fontSize: 13,
  },
  pipelineStepWrapper: { display: "flex", flexDirection: "column", gap: 0, width: "100%" },
  pipelineSummaryRow: { display: "flex", alignItems: "center", gap: 6, width: "100%" },
  pipelineSummaryToggle: {
    width: 22,
    height: 22,
    borderRadius: 4,
    border: "none",
    background: "rgba(15, 23, 42, 0.2)",
    color: "#cbd5f5",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    flexShrink: 0,
    outline: "none",
    transition: "background 0.2s ease, border-color 0.2s ease, color 0.2s ease, transform 0.2s ease",
  },
  pipelineSummaryToggleActive: {
    background: "rgba(59, 130, 246, 0.18)",
  },
  pipelineChevronIcon: { width: 12, height: 12, color: "inherit", transition: "transform 0.2s ease" },
  pipelineChevronPlaceholder: { width: 10, height: 2, borderRadius: 999, background: "rgba(148, 163, 184, 0.45)", display: "block" },
  pipelineSummaryText: {
    flex: 1,
    minWidth: 0,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
    fontSize: 13,
  },
  errorBubble: { alignSelf: "flex-start", background: "rgba(252, 165, 165, 0.32)", borderRadius: 22, padding: 15, maxWidth: "95%", boxShadow: "0 16px 28px rgba(239, 68, 68, 0.35)" },
  messageRole: { fontSize: 11, textTransform: "uppercase", letterSpacing: 0.8, color: "#ffffff", marginBottom: 2, whiteSpace: "nowrap" },
  sourcesBlock: { fontSize: 12, color: "#ffffff", marginTop: 10 },
  sourceItem: { marginBottom: 6, paddingBottom: 6, borderBottom: "1px solid rgba(148, 163, 184, 0.08)" },
  sourceHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" },
  sourceToggle: { font: "inherit", fontSize: 11, padding: "6px 16px", borderRadius: 999, border: "none", background: "rgba(65, 77, 128, 0.96)", color: "#ffffff", cursor: "pointer", boxShadow: "0 14px 24px rgba(3, 6, 18, 0.55)" },
  sourcePreview: {
    fontSize: 13,
    color: "#f8fbff",
    marginTop: 8,
    padding: "12px 14px",
    borderRadius: 16,
    border: "1px solid rgba(148, 163, 184, 0.2)",
    background: "rgba(15, 23, 42, 0.55)",
    lineHeight: 1.55,
    whiteSpace: "normal",
    wordBreak: "break-word",
  },
  stepDetailsPanel: {
    marginTop: 6,
    marginLeft: 28,
    padding: "8px 0 0 12px",
    borderLeft: "2px solid rgba(148, 163, 184, 0.3)",
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  stepDetailsHeaderRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    paddingBottom: 4,
  },
  stepDetailsHeaderTitle: { fontSize: 13, fontWeight: 600, color: "#f1f5f9", letterSpacing: 0.4 },
  stepDetailGroup: { display: "flex", flexDirection: "column", gap: 6 },
  stepDetailTitle: { fontSize: 11, color: "rgba(226, 232, 240, 0.85)", textTransform: "uppercase", letterSpacing: 0.8 },
  stepDetailList: { display: "flex", flexDirection: "column", gap: 6 },
  stepDetailRow: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    padding: "6px 8px",
    borderRadius: 10,
    background: "rgba(15, 23, 42, 0.35)",
  },
  stepDetailKey: { fontSize: 11, color: "rgba(226, 232, 240, 0.65)", textTransform: "uppercase", letterSpacing: 0.6 },
  stepDetailValue: { fontSize: 13, color: "#f8fbff", whiteSpace: "pre-wrap", wordBreak: "break-word" },
  stepDetailPre: {
    margin: 0,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
    background: "rgba(15, 23, 42, 0.65)",
    borderRadius: 8,
    padding: "6px 8px",
  },
  stepDetailsEmpty: { fontSize: 13, color: "rgba(226, 232, 240, 0.7)", fontStyle: "italic" },
  promptMessageList: { display: "flex", flexDirection: "column", gap: 6, marginTop: 6 },
  promptMessage: {
    borderRadius: 10,
    padding: "8px 10px",
    background: "rgba(14, 21, 40, 0.75)",
    border: "1px solid rgba(79, 70, 229, 0.25)",
  },
  promptRole: { fontSize: 11, color: "rgba(226, 232, 240, 0.8)", textTransform: "uppercase", letterSpacing: 0.6, marginBottom: 4 },
  promptContent: { fontSize: 13, color: "#f1f5f9", whiteSpace: "pre-wrap", wordBreak: "break-word" },
  toolCountText: { fontSize: 12, color: "rgba(226, 232, 240, 0.75)" },
  toolResultList: { display: "flex", flexDirection: "column", gap: 8 },
  toolResultCard: {
    borderRadius: 12,
    border: "1px solid rgba(148, 163, 184, 0.2)",
    padding: "10px 12px",
    background: "rgba(11, 15, 32, 0.65)",
    gap: 6,
    display: "flex",
    flexDirection: "column",
  },
  toolResultHeader: { display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" },
  toolResultTitle: { fontWeight: 600, color: "#e2e8f0", fontSize: 13 },
  toolResultScore: { fontSize: 12, color: "rgba(226, 232, 240, 0.7)" },
  toolResultMeta: { fontSize: 12, color: "rgba(226, 232, 240, 0.65)" },
  toolResultPreview: { fontSize: 13, color: "#f8fafc", whiteSpace: "pre-wrap", wordBreak: "break-word" },
  markdown: { fontSize: 14, lineHeight: 1.65, color: "#fbfcff", whiteSpace: "normal", wordBreak: "break-word" },
  markdownCompact: { fontSize: 13, lineHeight: 1.1, color: "#fbfcff", whiteSpace: "normal", wordBreak: "break-word", margin: 0, padding: 0 },
  markdownTable: { width: "100%", borderCollapse: "collapse", margin: "12px 0" },
  tableCell: { border: "1px solid rgba(148, 163, 184, 0.18)", padding: "8px 10px", textAlign: "left" },
  inlineCode: { background: "rgba(15, 23, 42, 0.6)", borderRadius: 8, padding: "2px 6px", fontSize: 13, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' },
  codeBlock: { background: "rgba(15, 23, 42, 0.75)", borderRadius: 16, padding: "14px 16px", margin: "12px 0", overflowX: "auto", fontSize: 13, border: "1px solid rgba(148, 163, 184, 0.25)" },
  contextBadge: { padding: "8px 14px", borderRadius: 14, border: "1px solid rgba(255, 255, 255, 0.6)", fontSize: 12, color: "#ffffff", background: "rgba(12, 14, 22, 0.85)" },
  contextLabel: { fontSize: 12, color: "#ffffff" },
  kbd: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' },
};

const markdownRemarkPlugins = [remarkGfm, remarkMath];
const markdownRehypePlugins = [rehypeRaw, rehypeKatex];
const markdownComponents = {
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
const pipelineMarkdownComponents = {
  ...markdownComponents,
  p: (props) => <p style={{ margin: 0 }}>{props.children}</p>,
};

export default function ChatPage({ onAskingChange, warmupApi, llmReady, systemStatus = {} }) {
  const defaultContextStats = useMemo(() => makeDefaultContextStats(), []);
  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [warmingUp, setWarmingUp] = useState(false);
  const [warmedUp, setWarmedUp] = useState(false);
  const [contextStats, setContextStats] = useState(() => makeDefaultContextStats());
  const [pendingFollowUp, setPendingFollowUp] = useState(null);
  const [continuing, setContinuing] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const [expandedStepDetails, setExpandedStepDetails] = useState({});
  const [activeDiagnosticsPanel, setActiveDiagnosticsPanel] = useState(null);
  const [matchesPanelOpen, setMatchesPanelOpen] = useState(false);
  const warmupAttemptRef = useRef(false);
  const messagesBodyRef = useRef(null);
  const queryInputRef = useRef(null);
  const navigate = useNavigate();

  const settingsGroups = useMemo(() => {
    const base = systemStatus?.settings;
    const merged = base ? { ...base } : {};
    if (systemStatus?.gpu_phase) {
      merged.gpu = {
        state: systemStatus.gpu_phase.state || "unknown",
        last_error: systemStatus.gpu_phase.last_error || "",
      };
    }
    return Object.keys(merged).length > 0 ? merged : null;
  }, [systemStatus]);

  const { data: gpuStats, error: gpuError, loading: gpuLoading } = useGpuDiagnostics(activeDiagnosticsPanel === "gpu");
  const handleDiagnosticsPanelChange = useCallback((panelKey) => setActiveDiagnosticsPanel(panelKey), []);
  const toggleMatchesPanel = useCallback(() => setMatchesPanelOpen((prev) => !prev), []);
  const latestMatches = useMemo(() => {
    for (let idx = messages.length - 1; idx >= 0; idx -= 1) {
      const msg = messages[idx];
      if (!msg || msg.role !== "assistant") continue;
      const retrieval = Array.isArray(msg.retrievalSources) ? msg.retrievalSources : [];
      if (retrieval.length > 0) return retrieval;
      const fallback = Array.isArray(msg.sources) ? msg.sources : [];
      if (fallback.length > 0) return fallback;
    }
    return [];
  }, [messages]);

  const api = { askStream: "/api/ask/agentic/stream" };
  useEffect(() => { if (onAskingChange) onAskingChange(asking || warmingUp || continuing); }, [asking, warmingUp, continuing, onAskingChange]);
  useEffect(() => {
    const el = messagesBodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if ((warmingUp && !warmedUp) || pendingFollowUp || continuing) return;
    const inputEl = queryInputRef.current;
    if (inputEl) inputEl.focus();
  }, [warmingUp, warmedUp, pendingFollowUp, continuing]);

  const performWarmup = useCallback(async () => {
    if (!warmupApi || warmedUp || llmReady) return; if (warmupAttemptRef.current) return; warmupAttemptRef.current = true; setWarmingUp(true);
    try { const res = await fetch(warmupApi, { method: "POST" }); const data = await readJsonSafe(res); if (res.ok && data.warmup_complete) { setWarmedUp(true); } else { warmupAttemptRef.current = false; } }
    catch (e) { warmupAttemptRef.current = false; }
    finally { setWarmingUp(false); }
  }, [warmupApi, warmedUp, llmReady]);

  useEffect(() => { if (!warmedUp && !llmReady && warmupApi) { void performWarmup(); } }, [warmedUp, llmReady, warmupApi, performWarmup]);
  useEffect(() => { if (llmReady) { setWarmedUp(true); setWarmingUp(false); } }, [llmReady]);

  const handleResetConversation = () => {
    setConversationId(null);
    setMessages([]);
    setContextStats({ ...defaultContextStats });
    setPendingFollowUp(null);
    setContinuing(false);
    setExpandedSources({});
    setExpandedStepDetails({});
  };

  const runStreamingCompletion = useCallback(
    async ({ payload, targetMessageId = null, anchorMessageId = null, baseContent = "", baseSources = [], baseRetrievalSources = [] }) => {
      const res = await fetch(api.askStream, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (!res.ok || !res.body) {
        const data = await readJsonSafe(res);
        throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText || "Request failed");
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = baseContent || "";
      let finalMeta = null;
      let mergedSources = Array.isArray(baseSources) ? baseSources : [];
      let mergedRetrievalSources = Array.isArray(baseRetrievalSources) ? baseRetrievalSources : [];
      let assistantId = targetMessageId;

      const ensureAssistant = () => {
        if (assistantId) return assistantId;
        const newId = createMessageId();
        assistantId = newId;
        setMessages((prev) => [...prev, { id: newId, role: "assistant", content: "", sources: mergedSources, retrievalSources: mergedRetrievalSources }]);
        return newId;
      };

      // Steps belong to a specific ask/turn; anchorKey keeps steps per request separate
      const anchorKey = anchorMessageId || targetMessageId || assistantId || "run";

      const upsertStepBubble = (stepInfo) => {
        if (!stepInfo) return;
        const key =
          stepInfo.order != null ? `order-${stepInfo.order}` : `${stepInfo.name || stepInfo.kind || "step"}-${stepInfo.kind || "unk"}`;
        const stepKey = `${anchorKey}-${key}`;
        const stepPayload = stepInfo ? { ...stepInfo } : null;
        const contentLine =
          stepInfo.state === "started"
            ? `${stepInfo.name || stepInfo.kind || "step"} (in progress...)`
            : formatSteps([stepInfo])[0] || stepInfo.name || stepInfo.kind || "step";
        const stepTypeLabel = stepInfo && stepInfo.kind ? (stepInfo.kind === "tool" ? "Tool" : "LLM") : null;
        const decoratedLine = stepTypeLabel ? `[${stepTypeLabel}] ${contentLine}` : contentLine;
        if (!contentLine) return;
        setMessages((prev) => {
          const next = [...prev];
          const anchorIdx = next.findIndex((m) => m.id === anchorMessageId || m.id === targetMessageId || m.id === assistantId);
          const lastPipelineIdx = next.reduce((acc, m, idx) => {
            const isPipeline = m.role === "system" && (m.title || "").toLowerCase() === "pipeline";
            if (!isPipeline) return acc;
            if (m.anchorKey && m.anchorKey === anchorKey) return Math.max(acc, idx);
            return acc;
          }, -1);
          const insertionIdx =
            lastPipelineIdx >= 0 && next[lastPipelineIdx]?.anchorKey === anchorKey
              ? lastPipelineIdx + 1
              : anchorIdx >= 0
              ? anchorIdx + 1
              : next.length;
          const existingIdx = next.findIndex((m) => m.stepKey === stepKey);
          if (existingIdx >= 0) {
            next[existingIdx] = { ...next[existingIdx], content: decoratedLine, state: stepInfo.state, isPipeline: true, stepInfo: stepPayload };
          } else {
            next.splice(insertionIdx, 0, {
              id: createMessageId(),
              role: "system",
              title: "Pipeline",
              isPipeline: true,
              content: decoratedLine,
              stepKey,
              state: stepInfo.state,
              anchorKey,
              stepInfo: stepPayload,
            });
          }
          return next;
        });
      };

      const updateAssistant = (content) => {
        const activeId = assistantId || targetMessageId;
        if (!activeId) return;
        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id !== activeId) return msg;
            return { ...msg, content };
          }),
        );
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const raw of lines) {
          const line = raw.trim();
          if (!line) continue;
          let evt;
          try {
            evt = JSON.parse(line);
          } catch {
            continue;
          }
          if (evt.type === "token") {
            ensureAssistant();
            const delta = evt.content ?? evt.token ?? "";
            if (!delta) continue;
            accumulated += delta;
            updateAssistant(accumulated);
          } else if (evt.type === "step") {
            const steps = Array.isArray(evt.step) ? evt.step : [evt.step];
            steps.filter(Boolean).forEach((s) => upsertStepBubble(s));
          } else if (evt.type === "final") {
            finalMeta = evt;
            if (typeof evt.answer === "string") {
              accumulated = evt.answer;
              updateAssistant(accumulated);
            }
            mergedSources = mergeSources(mergedSources, evt.sources);
            mergedRetrievalSources = mergeSources(mergedRetrievalSources, evt.retrieval_sources);
          } else if (evt.type === "error") {
            throw new Error(evt.error || "Streaming error");
          }
        }
      }

      const tail = buffer.trim();
      if (tail) {
        try {
          const evt = JSON.parse(tail);
          if (evt.type === "final") {
            finalMeta = evt;
            if (typeof evt.answer === "string") {
              accumulated = evt.answer;
              updateAssistant(accumulated);
            }
            mergedSources = mergeSources(mergedSources, evt.sources);
            mergedRetrievalSources = mergeSources(mergedRetrievalSources, evt.retrieval_sources);
          } else if (evt.type === "error") {
            throw new Error(evt.error || "Streaming error");
          }
        } catch {
          // Ignore trailing parse errors
        }
      }

      if (!finalMeta) throw new Error("Stream ended without a final payload");
      const nextConversationId = finalMeta.conversation_id || payload.conversation_id || conversationId;
      if (nextConversationId) setConversationId(nextConversationId);
      const needsFollowUp = !!finalMeta.needs_follow_up;
      const consolidatedSteps = Array.isArray(finalMeta.steps) ? finalMeta.steps : [];
      if (consolidatedSteps.length) {
        const sortedSteps = consolidatedSteps.slice().sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
        sortedSteps.forEach((s) => upsertStepBubble(s));
      }
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== (assistantId || targetMessageId)) return msg;
          return {
            ...msg,
            content: accumulated,
            sources: mergeSources(mergedSources, finalMeta.sources),
            retrievalSources: mergeSources(mergedRetrievalSources, finalMeta.retrieval_sources),
            hideSources: needsFollowUp,
            pendingFollowUp: needsFollowUp,
            finishReason: finalMeta.finish_reason || null,
            timing: {
              timeToFirst: finalMeta.time_to_first_token_seconds,
              generationSeconds: finalMeta.generation_seconds,
              tokensPerSecond: finalMeta.tokens_per_second,
            },
          };
        }),
      );
      setContextStats({
        used: typeof finalMeta.context_tokens_used === "number" ? finalMeta.context_tokens_used : 0,
        limit: typeof finalMeta.context_window_limit === "number" ? finalMeta.context_window_limit : defaultContextStats.limit,
        truncated: !!finalMeta.context_truncated,
        ratio: typeof finalMeta.context_usage === "number" ? finalMeta.context_usage : 0,
      });
      setPendingFollowUp(needsFollowUp ? { conversationId: nextConversationId, messageId: assistantId || targetMessageId } : null);
      // When streaming, steps are emitted individually via step events; final steps (if any) are also appended above.
    },
    [api.askStream, conversationId, defaultContextStats.limit],
  );

  const handleAsk = async () => {
    const trimmed = query.trim();
    if (!trimmed || (warmingUp && !warmedUp) || pendingFollowUp || continuing) return;
    setAsking(true);
    const userId = createMessageId();
    setMessages((prev) => [...prev, { id: userId, role: "user", content: trimmed }]);
    setQuery("");
    try {
      const payload = { query: trimmed };
      if (conversationId) payload.conversation_id = conversationId;
      await runStreamingCompletion({ payload, anchorMessageId: userId });
    } catch (e) {
      setMessages((prev) => [...prev, { id: createMessageId(), role: "assistant", content: `Error: ${e.message || String(e)}`, error: true }]);
    } finally {
      setAsking(false);
    }
  };

  const handleContinueResponse = async () => {
    if (!pendingFollowUp || continuing) return;
    const activeConversationId = pendingFollowUp.conversationId || conversationId;
    if (!activeConversationId) return;
    const existingAssistant = messages.find((m) => m.id === pendingFollowUp.messageId) || {};
    setContinuing(true);
    try {
      const payload = { continue_last: true, conversation_id: activeConversationId };
      await runStreamingCompletion({
        payload,
        targetMessageId: pendingFollowUp.messageId,
        anchorMessageId: pendingFollowUp.messageId,
        baseContent: existingAssistant.content || "",
        baseSources: existingAssistant.sources || [],
        baseRetrievalSources: existingAssistant.retrievalSources || [],
      });
    } catch (e) {
      setMessages((prev) => [...prev, { id: createMessageId(), role: "assistant", content: `Error continuing response: ${e.message || String(e)}`, error: true }]);
    } finally {
      setContinuing(false);
    }
  };

  const handleAbortContinuation = () => {
    if (!pendingFollowUp || continuing) return;
    setMessages((prev) => prev.map((msg) => (msg.id === pendingFollowUp.messageId ? { ...msg, pendingFollowUp: false, hideSources: false, aborted: true } : msg)));
    setPendingFollowUp(null);
  };

  const toggleSourcePreview = (messageId, sourceKey) => {
    if (!messageId || !sourceKey) return;
    setExpandedSources((prev) => {
      const current = new Set(prev[messageId] || []);
      if (current.has(sourceKey)) {
        current.delete(sourceKey);
      } else {
        current.add(sourceKey);
      }
      const next = { ...prev };
      if (current.size === 0) {
        delete next[messageId];
      } else {
        next[messageId] = Array.from(current);
      }
      return next;
    });
  };

  const toggleStepDetails = useCallback((messageId) => {
    if (!messageId) return;
    setExpandedStepDetails((prev) => ({ ...prev, [messageId]: !prev[messageId] }));
  }, []);

  return (
    <>
      <DiagnosticsPanel
        activePanel={activeDiagnosticsPanel}
        onToggle={handleDiagnosticsPanelChange}
        groups={settingsGroups}
        gpu={gpuStats}
        gpuError={gpuError}
        gpuLoading={gpuLoading}
      />
      <RetrievalPanel open={matchesPanelOpen} onToggle={toggleMatchesPanel} sources={latestMatches} />
      <div style={styles.page}>
        <section style={styles.chatCard}>
        <div style={styles.sectionHeader}>
          <h2 style={styles.sectionTitle}>Chat Workspace</h2>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <button onClick={() => navigate("/ingest")} style={{ ...styles.subtleButton, padding: "8px 16px" }}>Back to Ingestion</button>
            <button onClick={handleResetConversation} disabled={!messages.length && !conversationId} style={{ ...styles.subtleButton, padding: "8px 16px", opacity: (!messages.length && !conversationId) ? 0.5 : 1 }}>Reset Chat</button>
            <ContextIndicator stats={contextStats} defaultLimit={ENV_CONTEXT_LIMIT} />
          </div>
        </div>

        {warmingUp && !llmReady && (
          <div style={{ ...styles.muted, background: "rgba(250, 204, 21, 0.22)", padding: 14, borderRadius: 18, boxShadow: "0 16px 32px rgba(251, 191, 36, 0.35)" }}>
            🔥 Warming up the local LLM... Please wait.
          </div>
        )}

        <div style={styles.messages} ref={messagesBodyRef}>
          {warmingUp && !warmedUp ? (
            <div style={styles.muted}>🔥 Initializing the local LLM for first use. This may take a moment...</div>
          ) : messages.length === 0 ? (
            <div style={styles.muted}>{warmedUp ? "✅ Ready! Ask a question to get started." : "Ask a question to get started."}</div>
          ) : (
            <div style={styles.messageList}>
              {messages.map((m, i) => {
                const expandedForMessage = expandedSources[m.id] || [];
                const isPipeline = m.isPipeline || (m.role === "system" && (m.title || "").toLowerCase() === "pipeline");
                const stepInfo = isPipeline ? m.stepInfo : null;
                const hasStepMetadata = !!(stepInfo && (stepInfo.error || (stepInfo.details && Object.keys(stepInfo.details).length > 0)));
                const stepDetailsExpanded = !!expandedStepDetails[m.id];
                const pipelineToggleTitle = hasStepMetadata
                  ? stepDetailsExpanded
                    ? "Hide step details"
                    : "Show step details"
                  : "No structured details for this step";
                const pipelineToggleStyle = {
                  ...styles.pipelineSummaryToggle,
                  ...(stepDetailsExpanded ? styles.pipelineSummaryToggleActive : {}),
                  cursor: hasStepMetadata ? "pointer" : "default",
                  opacity: hasStepMetadata ? 1 : 0.5,
                };
                const bubbleStyle = m.error
                  ? styles.errorBubble
                  : m.role === "user"
                  ? styles.userBubble
                  : m.role === "system" && isPipeline
                  ? styles.pipelineBubble
                  : m.role === "system"
                  ? styles.systemBubble
                  : styles.assistantBubble;
                const baseMarkdownStyle = isPipeline ? styles.markdownCompact : styles.markdown;
                const isChatParticipant = m.role === "user" || m.role === "assistant";
                const markdownStyle =
                  !isPipeline && isChatParticipant && typeof baseMarkdownStyle.lineHeight === "number"
                    ? { ...baseMarkdownStyle, lineHeight: baseMarkdownStyle.lineHeight * 0.7 }
                    : baseMarkdownStyle;
                const markdownRenderer = isPipeline ? pipelineMarkdownComponents : markdownComponents;
                const roleLabel = isChatParticipant ? "" : isPipeline ? "" : m.title ? m.title : "";
                return (
                  <div key={m.id || `${m.role}-${i}-${Math.abs(m.content?.length || 0)}`} style={bubbleStyle}>
                  {roleLabel ? <div style={styles.messageRole}>{roleLabel}</div> : null}
                  {isPipeline ? (
                    <div style={styles.pipelineStepWrapper}>
                      <div style={styles.pipelineSummaryRow}>
                        <button
                          type="button"
                          title={pipelineToggleTitle}
                          aria-label={pipelineToggleTitle}
                          aria-expanded={hasStepMetadata ? stepDetailsExpanded : undefined}
                          onClick={hasStepMetadata ? () => toggleStepDetails(m.id) : undefined}
                          disabled={!hasStepMetadata}
                          style={{
                            ...pipelineToggleStyle,
                            cursor: hasStepMetadata ? "pointer" : "not-allowed",
                          }}
                        >
                          {hasStepMetadata ? (
                            <ChevronIcon expanded={stepDetailsExpanded} />
                          ) : (
                            <span style={styles.pipelineChevronPlaceholder} />
                          )}
                        </button>
                        <div
                          style={{
                            ...styles.pipelineSummaryText,
                            cursor: hasStepMetadata ? "pointer" : "default",
                          }}
                          role={hasStepMetadata ? "button" : undefined}
                          tabIndex={hasStepMetadata ? 0 : undefined}
                          onClick={hasStepMetadata ? () => toggleStepDetails(m.id) : undefined}
                          onKeyDown={
                            hasStepMetadata
                              ? (evt) => {
                                  if (evt.key === "Enter" || evt.key === " ") {
                                    evt.preventDefault();
                                    toggleStepDetails(m.id);
                                  }
                                }
                              : undefined
                          }
                          title={hasStepMetadata ? pipelineToggleTitle : undefined}
                          aria-expanded={hasStepMetadata ? stepDetailsExpanded : undefined}
                        >
                          <div style={markdownStyle}>
                            <ReactMarkdown
                              remarkPlugins={markdownRemarkPlugins}
                              rehypePlugins={markdownRehypePlugins}
                              components={markdownRenderer}
                            >
                              {m.content || ""}
                            </ReactMarkdown>
                          </div>
                        </div>
                      </div>
                      {hasStepMetadata && stepDetailsExpanded ? (
                        <div style={styles.stepDetailsPanel}>
                          <StepDetails stepInfo={stepInfo} />
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div style={markdownStyle}>
                      <ReactMarkdown
                        remarkPlugins={markdownRemarkPlugins}
                        rehypePlugins={markdownRehypePlugins}
                        components={markdownRenderer}
                      >
                        {m.content || ""}
                      </ReactMarkdown>
                    </div>
                  )}
                  {m.role === "assistant" && m.pendingFollowUp && !m.error && (
                    <div style={{ ...styles.muted, marginTop: 8 }}>
                      Response paused because it reached the token limit. Continue or abort to proceed.
                    </div>
                  )}
                  {m.role === "assistant" && m.aborted && !m.error && (
                    <div style={{ ...styles.muted, marginTop: 8 }}>Generation aborted. You can ask another question.</div>
                  )}
                  {m.role === "assistant" && m.timing && (
                    <div style={{ ...styles.muted, marginTop: 8, fontSize: 12 }}>
                      {formatTiming(m.timing)}
                    </div>
                  )}
                  {m.role === "assistant" && Array.isArray(m.sources) && m.sources.length > 0 && !m.hideSources && (
                    <div style={styles.sourcesBlock}>
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>Sources</div>
                      <ol style={{ margin: 0, paddingLeft: 18 }}>
                        {m.sources.map((s, idx) => {
                          const sourceKey = `${s.chunk_id || s.doc_hash || idx}-${idx}`;
                          const isExpanded = expandedForMessage.includes(sourceKey);
                          const fullChunkText = typeof s.chunk_text === "string" && s.chunk_text.length ? s.chunk_text : "";
                          const fallbackPreview = typeof s.chunk_text_preview === "string" ? s.chunk_text_preview : "";
                          const chunkText = fullChunkText || fallbackPreview;
                          const hasChunkText = chunkText.length > 0;
                          const previewOnly = !fullChunkText && !!fallbackPreview;
                          const chunkDisplayText = previewOnly ? `${chunkText}...` : chunkText;
                          return (
                            <li key={sourceKey} style={styles.sourceItem}>
                              <div style={styles.sourceHeader}>
                                <div>
                                  <strong style={{ color: "rgba(226, 232, 240, 0.95)" }}>{s.document_name || "unknown"}</strong>
                                  {s.total_chunks > 0 && (
                                    <span style={{ marginLeft: 8, fontSize: 12 }}>
                                      chunk {s.order_index + 1}/{s.total_chunks}
                                    </span>
                                  )}
                                  {typeof s.score === "number" && (
                                    <span style={{ marginLeft: 8, fontSize: 12, color: "rgba(148, 163, 184, 0.7)" }}>
                                      similarity: {(s.score * 100).toFixed(1)}%
                                    </span>
                                  )}
                                </div>
                                {hasChunkText && (
                                  <button type="button" onClick={() => toggleSourcePreview(m.id, sourceKey)} style={{ ...styles.sourceToggle, opacity: isExpanded ? 0.85 : 1 }}>
                                    {isExpanded ? "Hide chunk" : "Show chunk"}
                                  </button>
                                )}
                              </div>
                              {isExpanded && hasChunkText && (
                                <div style={styles.sourcePreview}>
                                  <ReactMarkdown
                                    remarkPlugins={markdownRemarkPlugins}
                                    rehypePlugins={markdownRehypePlugins}
                                    components={markdownComponents}
                                  >
                                    {chunkDisplayText}
                                  </ReactMarkdown>
                                </div>
                              )}
                            </li>
                          );
                        })}
                      </ol>
                    </div>
                  )}
                </div>
                );
              })}
            </div>
          )}
        </div>

        {pendingFollowUp && (
          <div style={{ border: "none", borderRadius: 22, padding: 16, background: "rgba(32, 37, 78, 0.92)", display: "flex", flexDirection: "column", gap: 12, boxShadow: "0 22px 40px rgba(4, 7, 20, 0.55)" }}>
            <div style={{ ...styles.muted, fontSize: 13 }}>
              The assistant stopped early (finish reason: {messages.find((m) => m.id === pendingFollowUp.messageId)?.finishReason || "unknown"}).
              Choose <strong>Continue</strong> to keep generating or <strong>Abort</strong> to accept the current response.
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={handleAbortContinuation} disabled={continuing} style={{ ...styles.subtleButton, padding: "10px 18px", opacity: continuing ? 0.6 : 1 }}>
                Abort
              </button>
              <button onClick={handleContinueResponse} disabled={continuing} style={{ ...styles.button, opacity: continuing ? 0.6 : 1 }}>
                {continuing ? "Continuing..." : "Continue"}
              </button>
            </div>
          </div>
        )}

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <input ref={queryInputRef} type="text" placeholder={warmingUp && !warmedUp ? "Warming up model..." : "Ask a question about your docs..."} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !(warmingUp && !warmedUp)) handleAsk(); }} disabled={(warmingUp && !warmedUp) || pendingFollowUp || continuing} style={{ ...styles.input, opacity: (warmingUp && !warmedUp) || pendingFollowUp || continuing ? 0.6 : 1 }} />
          <button onClick={handleAsk} disabled={asking || !query.trim() || (warmingUp && !warmedUp) || pendingFollowUp || continuing} style={{ ...styles.button, minWidth: 70, letterSpacing: 0.3, opacity: (asking || !query.trim() || (warmingUp && !warmedUp) || pendingFollowUp || continuing) ? 0.6 : 1 }}>
            {asking ? "Asking..." : warmingUp && !warmedUp ? "Warming up..." : "Ask"}
          </button>
        </div>
      </section>

    </div>
    </>
  );
}

function ChevronIcon({ expanded }) {
  return (
    <svg
      viewBox="0 0 24 24"
      role="presentation"
      aria-hidden="true"
      style={{ ...styles.pipelineChevronIcon, transform: expanded ? "rotate(180deg)" : "rotate(0deg)" }}
    >
      <polyline
        points="6 9 12 15 18 9"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ContextIndicator({ stats, defaultLimit }) {
  const limit = stats?.limit || defaultLimit || 10000;
  const used = stats?.used || 0;
  const percent = limit ? Math.min(100, Math.round((used / limit) * 100)) : 0;
  const accent = percent >= 85 ? "rgba(248, 113, 113, 0.95)" : percent >= 60 ? "rgba(251, 191, 36, 0.95)" : "rgba(16, 185, 129, 0.95)";
  const badgeStyle = {
    ...styles.contextBadge,
    borderColor: accent,
    color: accent,
    background: stats?.truncated ? "rgba(127, 29, 29, 0.25)" : styles.contextBadge.background,
    boxShadow: stats?.truncated ? "0 0 18px rgba(248, 113, 113, 0.55)" : "none",
  };
  return (
    <span style={badgeStyle} title="Prompt tokens used / max window">
      Context {used}/{limit} ({percent}%)
    </span>
  );
}

function StepDetails({ stepInfo }) {
  if (!stepInfo) return null;
  const rawDetails = stepInfo.details ? { ...stepInfo.details } : {};
  const isToolStep = stepInfo.kind === "tool";
  const llmPromptMessages = rawDetails.llm_prompt_messages;
  const llmPrompt = rawDetails.llm_prompt;
  const llmResponse = rawDetails.llm_response;
  const toolArgs = rawDetails.tool_args ?? rawDetails.args;
  const toolReturnCount = rawDetails.tool_return_count ?? rawDetails.results_count;
  const toolResults = rawDetails.tool_results ?? rawDetails.results;
  delete rawDetails.llm_prompt_messages;
  delete rawDetails.llm_prompt;
  delete rawDetails.llm_response;
  delete rawDetails.tool_args;
  delete rawDetails.tool_return_count;
  delete rawDetails.results_count;
  delete rawDetails.tool_results;
  delete rawDetails.results;
  delete rawDetails.args;

  const toolParamRows = toolArgs && typeof toolArgs === "object" && !Array.isArray(toolArgs)
    ? Object.entries(toolArgs).map(([label, value]) => [humanizeDetailKey(label), value])
    : toolArgs != null
    ? [["Value", toolArgs]]
    : [];

  const { inputs, outputs } = categorizeStepDetails(rawDetails);
  if (stepInfo.error) outputs.push(["Error", stepInfo.error]);
  const hasInputs = inputs.length > 0;
  const hasOutputs = outputs.length > 0;
  const hasLlmDetails = stepInfo.kind !== "tool" && ((llmPromptMessages && llmPromptMessages.length > 0) || llmPrompt || llmResponse);
  const hasToolDetails = isToolStep && (toolParamRows.length > 0 || (Array.isArray(toolResults) && toolResults.length > 0) || typeof toolReturnCount === "number");
  const hasGeneral = hasInputs || hasOutputs;
  if (!hasGeneral && !hasLlmDetails && !hasToolDetails) {
    return <div style={styles.stepDetailsEmpty}>No structured details captured for this step.</div>;
  }

  const typeLabel = stepInfo && stepInfo.kind ? (stepInfo.kind === "tool" ? "Tool" : "LLM") : "";
  const formattedLabels = formatSteps([stepInfo]);
  const formattedName = Array.isArray(formattedLabels)
    ? (formattedLabels[0] || "")
    : formattedLabels || "";
  const safeName = formattedName || stepInfo.name || stepInfo.kind || "Step";
  const headerLabel = typeLabel ? `[${typeLabel.toUpperCase()}] ${safeName}` : safeName;

  return (
    <>
      <div style={styles.stepDetailsHeaderRow}>
        <div style={styles.stepDetailsHeaderTitle}>{headerLabel}</div>
      </div>
      {hasLlmDetails ? (
        <div style={styles.stepDetailGroup}>
          <div style={styles.stepDetailTitle}>LLM Prompt</div>
          {llmPrompt ? <pre style={styles.stepDetailPre}>{llmPrompt}</pre> : null}
          {Array.isArray(llmPromptMessages) && llmPromptMessages.length > 0 ? (
            <div style={styles.promptMessageList}>
              {llmPromptMessages.map((msg, idx) => (
                <div style={styles.promptMessage} key={`${msg?.role || "message"}-${idx}`}>
                  <div style={styles.promptRole}>{(msg?.role || `Message ${idx + 1}`).toUpperCase()}</div>
                  <div style={styles.promptContent}>{msg?.content || ""}</div>
                </div>
              ))}
            </div>
          ) : null}
          {llmResponse ? (
            <div style={{ marginTop: 8 }}>
              <div style={styles.stepDetailTitle}>LLM Response</div>
              <pre style={styles.stepDetailPre}>{llmResponse}</pre>
            </div>
          ) : null}
        </div>
      ) : null}

      {hasToolDetails ? (
        <>
          {toolParamRows.length > 0 ? <StepDetailGroup title="Tool Parameters" rows={toolParamRows} /> : null}
          {typeof toolReturnCount === "number" ? <div style={styles.toolCountText}>Matches returned: {toolReturnCount}</div> : null}
          {Array.isArray(toolResults) && toolResults.length > 0 ? (
            <div style={styles.stepDetailGroup}>
              <div style={styles.stepDetailTitle}>Tool Output</div>
              <div style={styles.toolResultList}>
                {toolResults.map((res, idx) => {
                  const metaParts = [];
                  if (res?.chunk_id) {
                    const idxLabel = typeof res?.order_index === "number" ? res.order_index + 1 : null;
                    metaParts.push(idxLabel ? `chunk ${idxLabel}` : `chunk ${res.chunk_id}`);
                  } else if (typeof res?.order_index === "number") {
                    metaParts.push(`chunk ${res.order_index + 1}`);
                  }
                  if (res?.match_type) {
                    metaParts.push(res.match_type);
                  }
                  return (
                    <div key={`${res?.chunk_id || res?.document_name || idx}-${idx}`} style={styles.toolResultCard}>
                      <div style={styles.toolResultHeader}>
                        <span style={styles.toolResultTitle}>{res?.document_name || "Unknown"}</span>
                        {typeof res?.score === "number" ? (
                          <span style={styles.toolResultScore}>score {(res.score * 100).toFixed(1)}%</span>
                        ) : null}
                      </div>
                    {metaParts.length ? <div style={styles.toolResultMeta}>{metaParts.join(" · ")}</div> : null}
                    {res?.preview ? <div style={styles.toolResultPreview}>{res.preview}</div> : null}
                  </div>
                );
                })}
              </div>
            </div>
          ) : null}
        </>
      ) : null}

      {hasInputs ? <StepDetailGroup title="Inputs" rows={inputs} /> : null}
      {hasOutputs ? <StepDetailGroup title="Outputs" rows={outputs} /> : null}
    </>
  );
}

function StepDetailGroup({ title, rows }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div style={styles.stepDetailGroup}>
      <div style={styles.stepDetailTitle}>{title}</div>
      <div style={styles.stepDetailList}>
        {rows.map(([label, value], idx) => {
          const formatted = formatStepDetailValue(value);
          const isMultiline = typeof formatted === "string" && formatted.includes("\n");
          return (
            <div key={`${label || "field"}-${idx}`} style={styles.stepDetailRow}>
              <div style={styles.stepDetailKey}>{label || "Field"}</div>
              <div style={styles.stepDetailValue}>{isMultiline ? <pre style={styles.stepDetailPre}>{formatted}</pre> : formatted}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
