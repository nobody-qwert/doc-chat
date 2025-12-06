import React, { useMemo, useState, useEffect } from "react";

const PANEL_WIDTH = 630;
const HANDLE_WIDTH = 34;

function formatPercent(score) {
  if (typeof score !== "number" || !Number.isFinite(score)) return "—";
  return `${(score * 100).toFixed(1)}%`;
}

function buildDocSummaries(sources = []) {
  const summaries = new Map();
  sources.forEach((src, idx) => {
    if (!src) return;
    const docKey = src.doc_hash || src.document_name || `doc-${idx}`;
    if (!summaries.has(docKey)) {
      summaries.set(docKey, {
        key: docKey,
        name: src.document_name || "Document",
        chunkCount: 0,
        maxScore: Number.NEGATIVE_INFINITY,
      });
    }
    const entry = summaries.get(docKey);
    entry.chunkCount += 1;
    if (typeof src.score === "number" && src.score > entry.maxScore) {
      entry.maxScore = src.score;
    }
  });
  return Array.from(summaries.values())
    .map((entry) => ({
      ...entry,
      maxScore: Number.isFinite(entry.maxScore) ? entry.maxScore : null,
    }))
    .sort((a, b) => {
      const scoreA = Number.isFinite(a.maxScore) ? a.maxScore : -Infinity;
      const scoreB = Number.isFinite(b.maxScore) ? b.maxScore : -Infinity;
      return scoreB - scoreA;
    });
}

export default function RetrievalPanel({ open, onToggle, sources }) {
  const sourceList = Array.isArray(sources) ? sources : [];
  const docSummaries = useMemo(() => buildDocSummaries(sourceList), [sourceList]);
  const [expanded, setExpanded] = useState(() => new Set());

  useEffect(() => {
    setExpanded(new Set());
  }, [sourceList]);

  const toggleChunk = (chunkKey) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(chunkKey)) {
        next.delete(chunkKey);
      } else {
        next.add(chunkKey);
      }
      return next;
    });
  };

  const renderChunkPreview = (chunk) => {
    const fullText = typeof chunk?.chunk_text === "string" ? chunk.chunk_text : "";
    const preview = typeof chunk?.chunk_text_preview === "string" ? chunk.chunk_text_preview : "";
    const text = fullText || preview;
    if (!text) return null;
    return (
      <div
        style={{
          marginTop: 8,
          padding: "10px 12px",
          borderRadius: 14,
          background: "rgba(15,23,42,0.75)",
          color: "#e2e8f0",
          fontSize: 12,
          lineHeight: 1.45,
          whiteSpace: "pre-wrap",
        }}
      >
        {text}
      </div>
    );
  };

  const renderChunkList = () => (
    <div
      style={{
        borderRadius: 16,
        padding: "16px 18px",
        background: "rgba(23, 37, 84, 0.92)",
        boxShadow: "inset 0 0 0 1px rgba(59, 130, 246, 0.25)",
      }}
    >
      <div style={{ fontSize: 13, letterSpacing: 0.6, textTransform: "uppercase", color: "rgba(191, 219, 254, 0.85)", marginBottom: 10 }}>Matched Chunks</div>
      {sourceList.length === 0 ? (
        <div style={{ fontSize: 12, opacity: 0.75 }}>No retrieval matches yet.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {sourceList.map((chunk, idx) => {
            const chunkKey = chunk.chunk_id || `${chunk.doc_hash || "doc"}-${chunk.order_index ?? idx}`;
            const expandedState = expanded.has(chunkKey);
            const chunkIndexLabel = typeof chunk.order_index === "number" && typeof chunk.total_chunks === "number"
              ? `chunk ${chunk.order_index + 1}/${chunk.total_chunks}`
              : typeof chunk.order_index === "number"
              ? `chunk ${chunk.order_index + 1}`
              : null;
            return (
              <div
                key={chunkKey}
                style={{
                  borderRadius: 14,
                  padding: "12px 14px",
                  background: "rgba(15, 23, 42, 0.85)",
                  boxShadow: "0 8px 16px rgba(2, 6, 23, 0.45)",
                  border: "1px solid rgba(99, 102, 241, 0.25)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                  <div style={{ fontSize: 12 }}>
                    <div style={{ fontWeight: 600, color: "#e0f2fe" }}>{chunk.document_name || "Document"}</div>
                    <div style={{ opacity: 0.75 }}>
                      {chunkIndexLabel}
                      {chunkIndexLabel ? " · " : ""}Sim {formatPercent(chunk.score)}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => toggleChunk(chunkKey)}
                    style={{
                      font: "inherit",
                      fontSize: 11,
                      padding: "6px 14px",
                      borderRadius: 999,
                      border: "none",
                      background: expandedState ? "rgba(249, 115, 22, 0.85)" : "rgba(99, 102, 241, 0.9)",
                      color: "#f8fafc",
                      cursor: "pointer",
                      boxShadow: "0 10px 16px rgba(2, 6, 23, 0.4)",
                    }}
                  >
                    {expandedState ? "Hide chunk" : "Show chunk"}
                  </button>
                </div>
                {expandedState ? renderChunkPreview(chunk) : null}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const renderDocSummary = () => (
    <div
      style={{
        borderRadius: 16,
        padding: "16px 18px",
        background: "rgba(30, 41, 59, 0.9)",
        boxShadow: "inset 0 0 0 1px rgba(96, 165, 250, 0.2)",
        marginBottom: 16,
      }}
    >
      <div style={{ fontSize: 13, letterSpacing: 0.6, textTransform: "uppercase", color: "rgba(226, 232, 240, 0.85)", marginBottom: 10 }}>Matched Documents</div>
      {docSummaries.length === 0 ? (
        <div style={{ fontSize: 12, opacity: 0.75 }}>Documents with matches will appear here.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {docSummaries.map((doc) => (
            <div
              key={doc.key}
              style={{
                borderRadius: 12,
                padding: "10px 12px",
                background: "rgba(15, 23, 42, 0.85)",
                border: "1px solid rgba(148, 163, 184, 0.25)",
              }}
            >
              <div style={{ fontWeight: 600 }}>{doc.name}</div>
              <div style={{ fontSize: 12, opacity: 0.8 }}>
                {doc.chunkCount} chunk{doc.chunkCount === 1 ? "" : "s"} · Top similarity {formatPercent(doc.maxScore)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        bottom: 0,
        right: 0,
        width: PANEL_WIDTH + HANDLE_WIDTH,
        pointerEvents: "none",
        zIndex: 24,
      }}
    >
      <aside
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          right: 0,
          width: PANEL_WIDTH,
          background: "rgba(7, 12, 31, 0.96)",
          borderRadius: "20px 0 0 20px",
          border: "1px solid rgba(56, 189, 248, 0.28)",
          boxShadow: "0 30px 60px rgba(5, 6, 20, 0.65)",
          backdropFilter: "blur(10px)",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
          color: "#e2e8f0",
          transform: open ? "translateX(0)" : `translateX(${PANEL_WIDTH}px)`,
          transition: "transform 0.25s ease",
          overflow: "hidden",
          pointerEvents: open ? "auto" : "none",
        }}
      >
        <div style={{ padding: "22px 20px", height: "100%", overflowY: "auto", display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <div style={{ fontSize: 13, letterSpacing: 0.4, opacity: 0.75 }}>Vector Retrieval</div>
            <div style={{ fontSize: 18, fontWeight: 600 }}>Match Details</div>
          </div>
          {renderDocSummary()}
          {renderChunkList()}
        </div>
      </aside>
      <button
        type="button"
        aria-label={open ? "Hide matches" : "Show matches"}
        aria-expanded={open}
        onClick={typeof onToggle === "function" ? onToggle : undefined}
        style={{
          position: "absolute",
          top: "50%",
          right: open ? PANEL_WIDTH : 0,
          transform: "translateY(-50%)",
          width: HANDLE_WIDTH,
          height: 110,
          borderRadius: "12px 0 0 12px",
          border: "1px solid rgba(59, 130, 246, 0.45)",
          background: open
            ? "linear-gradient(180deg, rgba(59,130,246,0.95), rgba(14,165,233,0.95))"
            : "linear-gradient(180deg, rgba(59,130,246,0.85), rgba(14,165,233,0.75))",
          color: "#f8fafc",
          cursor: "pointer",
          boxShadow: "0 12px 22px rgba(6, 10, 30, 0.65)",
          pointerEvents: "auto",
          writingMode: "vertical-rl",
          textTransform: "uppercase",
          letterSpacing: 0.8,
          fontSize: 11,
          fontWeight: 600,
        }}
      >
        Matches
      </button>
    </div>
  );
}
