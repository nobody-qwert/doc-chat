import React, { useEffect, useMemo, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom";

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    try { return await res.json(); } catch {}
  }
  const raw = await res.text();
  return { nonJson: true, raw };
}

import IngestPage from "./IngestPage";
import ChatPage from "./ChatPage";

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();
  const [systemStatus, setSystemStatus] = useState({
    ready: false,
    has_running_jobs: false,
    running_jobs: [],
    total_jobs: 0,
    asking: false,
    docs_count: 0,
    total_docs: 0,
    jobs: [],
    llm_ready: false,
    gpu_phase: null,
    documents: [],
    settings: null,
  });

  const api = useMemo(() => ({ status: "/api/status", warmup: "/api/warmup" }), []);

  useEffect(() => {
    let interval;
    const checkStatus = async () => {
      try {
        // synthesize status from separate endpoints
        const [docsRes, readyRes] = await Promise.all([
          fetch("/api/documents"),
          fetch("/api/ready"),
        ]);
        const [docs, readyInfo] = await Promise.all([docsRes, readyRes].map(readJsonSafe));
        const docsArr = Array.isArray(docs) ? docs : [];
        const ready = !!(readyInfo && readyInfo.ready);

        // jobs listing is optional; try and ignore errors
        let jobs = [];
        let statusSettings = null;
        let llmReadyValue;
        let gpuPhaseValue;
        try {
          const r = await fetch("/api/status");
          const j = await readJsonSafe(r);
          if (r.ok && j) {
            if (Array.isArray(j.jobs)) jobs = j.jobs;
            if (j.settings && typeof j.settings === "object") statusSettings = j.settings;
            if (Object.prototype.hasOwnProperty.call(j, "llm_ready")) {
              llmReadyValue = j.llm_ready;
            }
            if (Object.prototype.hasOwnProperty.call(j, "gpu_phase")) {
              gpuPhaseValue = j.gpu_phase;
            }
          }
        } catch {}

        setSystemStatus(prev => ({
          ...prev,
          ready,
          has_running_jobs: jobs.some(x => x.status === "running" || x.status === "queued"),
          running_jobs: jobs.filter(x => x.status === "running" || x.status === "queued"),
          total_jobs: jobs.length,
          docs_count: docsArr.filter(d => String(d.status || "").toLowerCase() === "processed").length,
          total_docs: docsArr.length,
          jobs,
          documents: docsArr,
          settings: statusSettings || prev.settings,
          llm_ready: llmReadyValue !== undefined ? llmReadyValue : prev.llm_ready,
          gpu_phase: gpuPhaseValue !== undefined ? gpuPhaseValue : prev.gpu_phase,
        }));
      } catch (e) {
        console.error("status error", e);
      }
    };
    checkStatus();
    interval = setInterval(checkStatus, 2000);
    return () => interval && clearInterval(interval);
  }, [api.status]);

  useEffect(() => {
    if (location.pathname === "/chat" && (!systemStatus.ready || systemStatus.has_running_jobs || systemStatus.docs_count === 0)) {
      navigate("/ingest", { replace: true });
    }
  }, [location.pathname, systemStatus.ready, systemStatus.has_running_jobs, systemStatus.docs_count, navigate]);

  const updateAskingStatus = (asking) => setSystemStatus(prev => ({ ...prev, asking }));

  return (
    <div style={styles.appShell}>
      <main style={styles.main}>
        <div style={styles.content}>
          <Routes>
            <Route path="/ingest" element={<IngestPage systemStatus={systemStatus} />} />
            <Route path="/chat" element={systemStatus.ready && systemStatus.docs_count > 0 && !systemStatus.has_running_jobs ? (
              <ChatPage
                onAskingChange={updateAskingStatus}
                warmupApi={api.warmup}
                llmReady={systemStatus.llm_ready}
                documents={systemStatus.documents}
                systemStatus={systemStatus}
              />
            ) : (<Navigate to="/ingest" replace />)} />
            <Route path="/" element={<Navigate to="/ingest" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

const styles = {
  appShell: { minHeight: "100vh", display: "flex", flexDirection: "column", background: "radial-gradient(circle at top left, rgba(45,55,95,0.35), rgba(15,17,23,0.98) 55%)", color: "#f4f6fb" },
  main: { flex: 1, padding: "16px 16px 16px" },
  content: { maxWidth: "1200px", margin: "0 auto", width: "100%", padding: "0 4px" },
};
