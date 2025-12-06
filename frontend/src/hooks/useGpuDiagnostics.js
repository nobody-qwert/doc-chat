import { useEffect, useState, useRef } from "react";

function parseJsonSafe(res) {
  const contentType = (res.headers.get("content-type") || "").toLowerCase();
  if (contentType.includes("application/json")) {
    return res.json().catch(() => ({}));
  }
  return res.text().then((raw) => ({ raw }));
}

export default function useGpuDiagnostics(active, intervalMs = 2500) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef({ abort: false });
  const hasSnapshotRef = useRef(false);

  useEffect(() => {
    controllerRef.current.abort = false;
    if (!active) {
      return () => { controllerRef.current.abort = true; };
    }

    let timer;

    const fetchSnapshot = async ({ silent = false } = {}) => {
      const showSpinner = !silent && !hasSnapshotRef.current;
      if (showSpinner) {
        setLoading(true);
      }
      try {
        const res = await fetch("/api/diagnostics/gpu", { cache: "no-store" });
        const payload = await parseJsonSafe(res);
        if (!res.ok) {
          throw new Error(payload?.detail || payload?.error || res.statusText);
        }
        if (!controllerRef.current.abort) {
          hasSnapshotRef.current = true;
          setData(payload);
          setError(payload?.error || null);
        }
      } catch (err) {
        if (!controllerRef.current.abort) {
          setError(err.message || String(err));
          if (!hasSnapshotRef.current) {
            setData(null);
          }
        }
      } finally {
        if (!controllerRef.current.abort && showSpinner) {
          setLoading(false);
        }
      }
    };

    fetchSnapshot();
    timer = setInterval(() => fetchSnapshot({ silent: true }), Math.max(1500, intervalMs));

    return () => {
      controllerRef.current.abort = true;
      if (timer) clearInterval(timer);
    };
  }, [active, intervalMs]);

  return { data, error, loading };
}
