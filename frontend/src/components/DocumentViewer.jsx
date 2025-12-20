import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist/build/pdf";
import PdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?worker";

if (!GlobalWorkerOptions.workerPort) {
  GlobalWorkerOptions.workerPort = new PdfWorker();
}

function PdfPage({ pdfDoc, pageNumber, scale }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let isCancelled = false;
    const renderPage = async () => {
      if (!pdfDoc || !canvasRef.current) return;
      const page = await pdfDoc.getPage(pageNumber);
      if (isCancelled || !canvasRef.current) return;
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = "100%";
      canvas.style.height = "auto";
      if (!context) return;
      const renderTask = page.render({ canvasContext: context, viewport });
      await renderTask.promise;
    };
    renderPage();
    return () => {
      isCancelled = true;
    };
  }, [pdfDoc, pageNumber, scale]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        borderRadius: 12,
        display: "block",
        background: "rgba(15, 23, 42, 0.6)",
        boxShadow: "0 12px 24px rgba(2, 6, 23, 0.45)",
      }}
    />
  );
}

export default function DocumentViewer({ docHash, docName, onClose }) {
  const [pdfDoc, setPdfDoc] = useState(null);
  const [numPages, setNumPages] = useState(0);
  const [pageTexts, setPageTexts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [matches, setMatches] = useState([]);
  const [activeMatch, setActiveMatch] = useState(0);
  const containerRef = useRef(null);
  const pageRefs = useRef([]);

  const fileUrl = useMemo(() => (docHash ? `/api/documents/${docHash}/file` : ""), [docHash]);

  useEffect(() => {
    if (!docHash) return;
    let isCancelled = false;
    setLoading(true);
    setError("");
    setPdfDoc(null);
    setMatches([]);
    setActiveMatch(0);
    setPageTexts([]);
    const loadPdf = async () => {
      try {
        const res = await fetch(fileUrl);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const contentType = res.headers.get("content-type") || "";
        if (!contentType.toLowerCase().includes("pdf")) {
          throw new Error("File is not a PDF");
        }
        const data = await res.arrayBuffer();
        const task = getDocument({ data });
        const doc = await task.promise;
        if (!doc) throw new Error("Empty PDF response");
        if (isCancelled) return;
        setPdfDoc(doc);
        setNumPages(doc.numPages || 0);
      } catch (err) {
        if (isCancelled) return;
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(`Unable to load the PDF file. (${message})`);
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };
    loadPdf();
    return () => {
      isCancelled = true;
    };
  }, [docHash, fileUrl]);

  useEffect(() => {
    if (!pdfDoc) return;
    let isCancelled = false;
    const loadText = async () => {
      setIndexing(true);
      const texts = Array(pdfDoc.numPages).fill("");
      for (let i = 1; i <= pdfDoc.numPages; i += 1) {
        if (isCancelled) return;
        const page = await pdfDoc.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items.map((item) => item.str).join(" ");
        texts[i - 1] = pageText;
      }
      if (!isCancelled) {
        setPageTexts(texts);
        setIndexing(false);
      }
    };
    loadText();
    return () => {
      isCancelled = true;
    };
  }, [pdfDoc]);

  const scrollToPage = useCallback((pageIndex) => {
    const el = pageRefs.current[pageIndex];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, []);

  const runSearch = useCallback(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) {
      setMatches([]);
      setActiveMatch(0);
      return;
    }
    const results = [];
    pageTexts.forEach((text, idx) => {
      if (!text) return;
      const haystack = text.toLowerCase();
      let count = 0;
      let pos = haystack.indexOf(query);
      while (pos !== -1) {
        count += 1;
        pos = haystack.indexOf(query, pos + query.length);
      }
      if (count > 0) results.push({ pageIndex: idx, count });
    });
    setMatches(results);
    setActiveMatch(0);
    if (results.length > 0) {
      scrollToPage(results[0].pageIndex);
    }
  }, [pageTexts, scrollToPage, searchQuery]);

  const goToMatch = useCallback(
    (direction) => {
      if (matches.length === 0) return;
      const nextIndex = (activeMatch + direction + matches.length) % matches.length;
      setActiveMatch(nextIndex);
      scrollToPage(matches[nextIndex].pageIndex);
    },
    [activeMatch, matches, scrollToPage],
  );

  const totalHits = matches.reduce((sum, entry) => sum + entry.count, 0);
  const matchLabel =
    matches.length > 0 ? `${activeMatch + 1}/${matches.length} pages · ${totalHits} hits` : "No matches";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        height: "100%",
        padding: "14px 16px 16px",
        background: "rgba(22, 27, 58, 0.98)",
        borderRadius: 22,
        boxShadow: "0 24px 48px rgba(3, 5, 15, 0.75)",
        border: "1px solid rgba(99, 102, 241, 0.2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: 0.6, textTransform: "uppercase", color: "rgba(226, 232, 240, 0.75)" }}>Document Viewer</div>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#f8fafc" }}>{docName || "Document"}</div>
        </div>
        <button
          type="button"
          onClick={onClose}
          style={{
            font: "inherit",
            fontSize: 12,
            padding: "6px 12px",
            borderRadius: 999,
            border: "none",
            background: "rgba(51, 65, 85, 0.85)",
            color: "#e2e8f0",
            cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          type="text"
          placeholder={indexing ? "Indexing text..." : "Search in document..."}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") runSearch();
          }}
          disabled={!pdfDoc || loading}
          style={{
            flex: 1,
            font: "inherit",
            fontSize: 13,
            padding: "8px 12px",
            borderRadius: 999,
            border: "1px solid rgba(59, 130, 246, 0.35)",
            background: "rgba(15, 23, 42, 0.7)",
            color: "#f8fafc",
            outline: "none",
          }}
        />
        <button
          type="button"
          onClick={runSearch}
          disabled={!pdfDoc || loading}
          style={{
            font: "inherit",
            fontSize: 12,
            padding: "7px 12px",
            borderRadius: 999,
            border: "none",
            background: "rgba(59, 130, 246, 0.9)",
            color: "#f8fafc",
            cursor: "pointer",
          }}
        >
          Find
        </button>
        <button
          type="button"
          onClick={() => goToMatch(-1)}
          disabled={matches.length === 0}
          style={{
            font: "inherit",
            fontSize: 12,
            padding: "7px 10px",
            borderRadius: 999,
            border: "none",
            background: "rgba(100, 116, 139, 0.8)",
            color: "#f8fafc",
            cursor: "pointer",
          }}
        >
          Prev
        </button>
        <button
          type="button"
          onClick={() => goToMatch(1)}
          disabled={matches.length === 0}
          style={{
            font: "inherit",
            fontSize: 12,
            padding: "7px 10px",
            borderRadius: 999,
            border: "none",
            background: "rgba(100, 116, 139, 0.8)",
            color: "#f8fafc",
            cursor: "pointer",
          }}
        >
          Next
        </button>
      </div>
      <div style={{ fontSize: 11, color: "rgba(226, 232, 240, 0.7)" }}>{matchLabel}</div>
      <div
        ref={containerRef}
        style={{
          flex: 1,
          overflowY: "auto",
          borderRadius: 18,
          padding: "10px 12px",
          background: "rgba(12, 16, 37, 0.75)",
          border: "1px solid rgba(59, 130, 246, 0.2)",
        }}
      >
        {loading ? (
          <div style={{ fontSize: 13, color: "rgba(226, 232, 240, 0.8)" }}>Loading PDF...</div>
        ) : error ? (
          <div style={{ fontSize: 13, color: "rgba(248, 113, 113, 0.9)" }}>{error}</div>
        ) : pdfDoc ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {Array.from({ length: numPages }, (_, idx) => {
              const pageNumber = idx + 1;
              return (
                <div key={`page-${pageNumber}`} ref={(el) => { pageRefs.current[idx] = el; }}>
                  <div style={{ fontSize: 11, color: "rgba(226, 232, 240, 0.65)", marginBottom: 6 }}>Page {pageNumber}</div>
                  <PdfPage pdfDoc={pdfDoc} pageNumber={pageNumber} scale={1.1} />
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ fontSize: 13, color: "rgba(226, 232, 240, 0.8)" }}>Select a document to view.</div>
        )}
      </div>
    </div>
  );
}
