import React, { useMemo } from "react";

const SECTION_LABELS = {
  ocr: "OCR",
  chunking: "Chunking",
  embedding: "Embeddings",
  llm: "LLM",
  retrieval: "Retrieval",
  storage: "Storage",
};

const PANEL_WIDTH = 420;
const HANDLE_WIDTH = 34;
const PANEL_KEYS = {
  GPU: "gpu",
  PARAMS: "params",
};

function clampPercent(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  return Math.min(100, Math.max(0, num));
}

const formatMiB = (value) => {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(0)} MiB`;
};

function normalizeGroups(groups) {
  if (!groups || typeof groups !== "object") return [];
  return Object.entries(groups)
    .map(([key, value]) => {
      const entries = value && typeof value === "object" ? Object.entries(value) : [];
      return {
        key,
        label: SECTION_LABELS[key] || key,
        entries,
      };
    })
    .filter((group) => group.entries.length > 0);
}

export default function DiagnosticsPanel({ activePanel, onToggle, groups, gpu, gpuError, gpuLoading }) {
  const sections = useMemo(() => normalizeGroups(groups), [groups]);
  const gpuList = Array.isArray(gpu?.gpus) ? gpu.gpus : [];
  const processList = Array.isArray(gpu?.processes) ? gpu.processes : [];
  const gpuMessage = gpuError || gpu?.error || null;
  const currentPanel = typeof activePanel === "string" ? activePanel : null;

  const isGpuOpen = currentPanel === PANEL_KEYS.GPU;
  const isParamsOpen = currentPanel === PANEL_KEYS.PARAMS;

  const handlePanelToggle = (panelKey) => {
    if (typeof onToggle !== "function") return;
    onToggle(currentPanel === panelKey ? null : panelKey);
  };

  const renderGpuSection = () => (
    <div
      style={{
        borderRadius: 16,
        padding: "14px 16px",
        marginBottom: 14,
        background: "rgba(19, 28, 66, 0.9)",
        boxShadow: "inset 0 0 0 1px rgba(94, 234, 212, 0.25)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontSize: 13, textTransform: "uppercase", letterSpacing: 0.8, color: "rgba(167, 243, 208, 0.85)" }}>GPU Usage</div>
        <div style={{ fontSize: 11, opacity: 0.65 }}>{gpu?.timestamp ? new Date(gpu.timestamp).toLocaleTimeString() : ""}</div>
      </div>
      {gpuLoading ? (
        <div style={{ fontSize: 12, opacity: 0.75 }}>Collecting metrics…</div>
      ) : gpuMessage ? (
        <div style={{ fontSize: 12, color: "#fda4af" }}>{gpuMessage}</div>
      ) : gpuList.length === 0 ? (
        <div style={{ fontSize: 12, opacity: 0.75 }}>No GPU metrics detected.</div>
      ) : (
        <>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {gpuList.map((gpuInfo) => {
              const used = Number(gpuInfo.memory_used_mb) || 0;
              const total = Number(gpuInfo.memory_total_mb) || 0;
              const percent = clampPercent(total > 0 ? (used / total) * 100 : 0);
              const gpuUtilValue = Number(gpuInfo.utilization_gpu);
              const gpuUtilPercent = Number.isFinite(gpuUtilValue) ? clampPercent(gpuUtilValue) : 0;
              const gpuUtilDisplay = Number.isFinite(gpuUtilValue) ? `${gpuUtilValue.toFixed(0)}%` : "—";
              return (
                <div key={`${gpuInfo.uuid}-${gpuInfo.index}`} style={{ background: "rgba(15, 23, 42, 0.75)", borderRadius: 12, padding: "10px 12px", boxShadow: "0 1px 0 rgba(148, 163, 184, 0.15)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
                    <strong style={{ color: "#e0f2fe" }}>GPU {gpuInfo.index}</strong>
                    <span style={{ opacity: 0.7 }}>{gpuInfo.name}</span>
                  </div>
                  <div style={{ fontSize: 12, marginBottom: 6 }}>Mem {formatMiB(used)} / {formatMiB(total)} ({percent.toFixed(0)}%)</div>
                  <div style={{ height: 6, borderRadius: 999, background: "rgba(148, 163, 184, 0.2)", overflow: "hidden", marginBottom: 6 }}>
                    <div style={{ width: `${percent}%`, height: "100%", background: "linear-gradient(90deg, rgba(14,165,233,0.9), rgba(79,70,229,0.9))" }} />
                  </div>
                  <div style={{ fontSize: 12, marginBottom: 6 }}>GPU Usage {gpuUtilDisplay}</div>
                  <div style={{ height: 6, borderRadius: 999, background: "rgba(148, 163, 184, 0.2)", overflow: "hidden", marginBottom: 6 }}>
                    <div
                      style={{
                        width: Number.isFinite(gpuUtilValue) ? `${gpuUtilPercent}%` : 0,
                        height: "100%",
                        background: "linear-gradient(90deg, rgba(249,115,22,0.9), rgba(236,72,153,0.9))",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: 0.6, opacity: 0.65, marginBottom: 6 }}>Active Processes</div>
            {processList.length === 0 ? (
              <div style={{ fontSize: 12, opacity: 0.7 }}>No compute processes.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 160, overflowY: "auto" }}>
                {processList.map((proc) => (
                  <div key={`${proc.pid}-${proc.name}-${proc.gpu_uuid}`} style={{ fontSize: 11, padding: "6px 8px", borderRadius: 10, background: "rgba(30, 41, 59, 0.75)", display: "grid", gridTemplateColumns: "60px 1fr 60px", gap: 8 }}>
                    <div style={{ opacity: 0.75 }}>GPU {proc.gpu_index ?? "?"}</div>
                    <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{proc.name || "process"} (PID {proc.pid})</div>
                    <div style={{ textAlign: "right", fontWeight: 600 }}>{formatMiB(proc.used_memory_mb)}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );

  const panelShellStyle = {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    width: PANEL_WIDTH,
    background: "rgba(11, 15, 34, 0.96)",
    borderRadius: "0 20px 20px 0",
    border: "1px solid rgba(99, 102, 241, 0.28)",
    boxShadow: "0 30px 60px rgba(5, 6, 20, 0.65)",
    backdropFilter: "blur(10px)",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
    color: "#cbd5f5",
    transition: "transform 0.25s ease",
    overflow: "hidden",
  };

  const renderParamsSection = () => (
    <div style={{ padding: "22px 20px", height: "100%", overflowY: "auto" }}>
      <div style={{ fontSize: 13, letterSpacing: 0.5, opacity: 0.75, marginBottom: 6 }}>Diagnostics</div>
      <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>Live Parameters</div>
      {sections.length === 0 ? (
        <div style={{ fontSize: 12, opacity: 0.7 }}>Settings unavailable.</div>
      ) : (
        sections.map((section) => (
          <div
            key={section.key}
            style={{
              borderRadius: 14,
              padding: "12px 14px",
              marginBottom: 12,
              background: "rgba(32, 38, 83, 0.85)",
              boxShadow: "inset 0 0 0 1px rgba(96, 165, 250, 0.18)",
            }}
          >
            <div style={{ fontSize: 13, textTransform: "uppercase", letterSpacing: 0.8, opacity: 0.75, marginBottom: 8 }}>{section.label}</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, fontVariantNumeric: "tabular-nums" }}>
              {section.entries.map(([name, value]) => (
                <div
                  key={name}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 12,
                    padding: "2px 0",
                    borderBottom: "1px dotted rgba(148, 163, 184, 0.18)",
                  }}
                >
                  <span style={{ opacity: 0.65 }}>{name}</span>
                  <span style={{ fontWeight: 500, wordBreak: "break-all", textAlign: "right" }}>
                    {value === undefined || value === null ? "—" : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );

  const renderGpuPanel = () => (
    <div style={{ padding: "22px 20px", height: "100%", overflowY: "auto" }}>
      <div style={{ fontSize: 13, letterSpacing: 0.5, opacity: 0.75, marginBottom: 6 }}>GPU Monitor</div>
      <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>Live Usage</div>
      {renderGpuSection()}
    </div>
  );

  const handleOffset = currentPanel ? PANEL_WIDTH : 0;
  const handleLabels = [
    { key: PANEL_KEYS.GPU, label: "GPU Usage" },
    { key: PANEL_KEYS.PARAMS, label: "Params" },
  ];

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        bottom: 0,
        left: 0,
        width: PANEL_WIDTH + HANDLE_WIDTH,
        pointerEvents: "none",
        zIndex: 20,
      }}
    >
      <aside
        style={{
          ...panelShellStyle,
          transform: isGpuOpen ? "translateX(0)" : `translateX(-${PANEL_WIDTH}px)`,
          pointerEvents: isGpuOpen ? "auto" : "none",
          zIndex: isGpuOpen ? 22 : 21,
        }}
      >
        {renderGpuPanel()}
      </aside>
      <aside
        style={{
          ...panelShellStyle,
          transform: isParamsOpen ? "translateX(0)" : `translateX(-${PANEL_WIDTH}px)`,
          pointerEvents: isParamsOpen ? "auto" : "none",
          zIndex: isParamsOpen ? 22 : 21,
        }}
      >
        {renderParamsSection()}
      </aside>
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: handleOffset,
          transform: "translateY(-50%)",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          pointerEvents: "auto",
        }}
      >
        {handleLabels.map(({ key, label }) => {
          const isActive = currentPanel === key;
          return (
            <button
              key={key}
              type="button"
              aria-label={isActive ? `Hide ${label}` : `Show ${label}`}
              aria-expanded={isActive}
              onClick={() => handlePanelToggle(key)}
              style={{
                width: HANDLE_WIDTH,
                height: 110,
                borderRadius: "0 12px 12px 0",
                border: "1px solid rgba(99, 102, 241, 0.4)",
                background: isActive
                  ? "linear-gradient(180deg, rgba(99,102,241,0.95), rgba(79,70,229,0.95))"
                  : "linear-gradient(180deg, rgba(79,70,229,0.85), rgba(56,189,248,0.85))",
                color: "#f8fafc",
                cursor: "pointer",
                boxShadow: "0 12px 22px rgba(6, 10, 30, 0.6)",
                writingMode: "vertical-rl",
                textTransform: "uppercase",
                letterSpacing: 0.8,
                fontSize: 11,
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "background 0.2s ease, opacity 0.2s ease",
                opacity: isActive ? 1 : 0.85,
              }}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
