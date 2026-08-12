import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Activity, AlertTriangle, CheckCircle2, ImageUp, Loader2, Server, Upload } from "lucide-react";
import React from "react";
import "./styles.css";

type Detection = {
  box_xyxy: [number, number, number, number];
  label: number;
  score: number;
};

type ScanRecord = {
  id: string;
  created_at: string;
  original_filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  prediction: {
    label: string;
    display_name: string;
    confidence: number;
    severity: string;
    is_mock: boolean;
    model_name: string;
    prediction_count: number;
    detections: Detection[];
  };
  evidence: {
    kind: string;
    summary: string;
  };
  report: {
    title: string;
    summary: string;
    limitations: string[];
    recommended_next_steps: string[];
    disclaimer: string;
  };
};

type ApiError = {
  detail?: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const SUPPORTED_IMAGE_EXTENSIONS: Record<string, readonly string[]> = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
};

function isSupportedImage(file: File) {
  const dotIndex = file.name.lastIndexOf(".");
  const extension = dotIndex >= 0 ? file.name.slice(dotIndex).toLowerCase() : "";
  return SUPPORTED_IMAGE_EXTENSIONS[file.type]?.includes(extension) ?? false;
}

function App() {
  const file = useFileState();
  const scan = useScanState();
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  async function submitScan() {
    if (!file.current) {
      setError("Select a JPEG or PNG image first.");
      fileInputRef.current?.focus();
      return;
    }
    setIsSubmitting(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file.current);
    try {
      const response = await fetch(`${API_BASE_URL}/scans`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => ({}))) as ApiError;
        throw new Error(payload.detail ?? "Scan request failed.");
      }
      scan.setCurrent((await response.json()) as ScanRecord);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Scan request failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="OralLens scan workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">OralLens AI</p>
            <h1>Screening support console</h1>
          </div>
          <div className="connection">
            <Server size={18} aria-hidden="true" />
            <span>{API_BASE_URL}</span>
          </div>
        </header>

        <div className="layout">
          <section className="upload-panel" aria-label="Image upload">
            <label className="dropzone">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png"
                aria-label="Choose oral image"
                aria-describedby="file-guidance"
                onChange={(event) => {
                  const selected = event.currentTarget.files?.[0] ?? null;
                  if (selected && !isSupportedImage(selected)) {
                    event.currentTarget.value = "";
                    file.setCurrent(null);
                    scan.setCurrent(null);
                    setError("Select a JPEG or PNG image.");
                    event.currentTarget.focus();
                    return;
                  }
                  file.setCurrent(selected);
                  scan.setCurrent(null);
                  setError(null);
                }}
              />
              <ImageUp size={28} aria-hidden="true" />
              <span>{file.current ? file.current.name : "Choose oral image"}</span>
              <small id="file-guidance">JPEG or PNG</small>
            </label>

            <div className="preview-frame">
              {file.previewUrl ? (
                <ImagePreview scan={scan.current} src={file.previewUrl} />
              ) : (
                <div className="empty-preview">
                  <Activity size={28} aria-hidden="true" />
                </div>
              )}
            </div>

            <button className="primary-action" type="button" onClick={submitScan} disabled={isSubmitting}>
              {isSubmitting ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <Upload size={18} aria-hidden="true" />}
              <span>{isSubmitting ? "Scanning" : "Run scan"}</span>
            </button>

            <p className="visually-hidden" role="status" aria-atomic="true">
              {isSubmitting ? "Scanning image." : scan.current ? "Scan complete." : ""}
            </p>

            {error ? (
              <div className="alert" role="alert">
                <AlertTriangle size={18} aria-hidden="true" />
                <span>{error}</span>
              </div>
            ) : null}
          </section>

          <ResultPanel scan={scan.current} />
        </div>
      </section>
    </main>
  );
}

function ImagePreview({ scan, src }: { scan: ScanRecord | null; src: string }) {
  const [size, setSize] = React.useState<{ width: number; height: number } | null>(null);
  const detections = scan?.prediction.detections ?? [];

  return (
    <div className="image-stage">
      <img
        src={src}
        alt="Selected oral scan"
        onLoad={(event) => {
          setSize({
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight,
          });
        }}
      />
      {size ? (
        <svg className="overlay" viewBox={`0 0 ${size.width} ${size.height}`} aria-hidden="true">
          {detections.map((detection, index) => {
            const [x1, y1, x2, y2] = detection.box_xyxy;
            return (
              <rect
                key={`${index}-${detection.score}`}
                x={x1}
                y={y1}
                width={Math.max(0, x2 - x1)}
                height={Math.max(0, y2 - y1)}
              />
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}

function ResultPanel({ scan }: { scan: ScanRecord | null }) {
  if (!scan) {
    return (
      <section className="result-panel empty-state" aria-label="Scan result">
        <CheckCircle2 size={28} aria-hidden="true" />
        <h2>Ready for a scan</h2>
        <p>Upload an oral image to view prediction metadata, evidence, and report text.</p>
      </section>
    );
  }

  const score = scan.prediction.confidence.toFixed(3);
  const scoreLabel = scan.prediction.is_mock ? "Mock score" : "Detector score";
  return (
    <section className="result-panel" aria-label="Scan result">
      <div className="result-heading">
        <div>
          <p className="eyebrow">{scan.prediction.is_mock ? "Mock adapter" : "MVP detector"}</p>
          <h2>{scan.prediction.display_name}</h2>
        </div>
        <div>
          <div className="score">{score}</div>
          <small>
            {scoreLabel}
            <br />
            not a clinical probability
          </small>
        </div>
      </div>

      <dl className="metrics-grid">
        <div>
          <dt>Model</dt>
          <dd>{scan.prediction.model_name}</dd>
        </div>
        <div>
          <dt>Severity</dt>
          <dd>{scan.prediction.severity}</dd>
        </div>
        <div>
          <dt>Detections</dt>
          <dd>{scan.prediction.prediction_count}</dd>
        </div>
        <div>
          <dt>Image size</dt>
          <dd>{formatBytes(scan.size_bytes)}</dd>
        </div>
      </dl>

      <section className="report-section">
        <h3>Evidence</h3>
        <p>{scan.evidence.summary}</p>
      </section>

      <section className="report-section">
        <h3>Report</h3>
        <p>{scan.report.summary}</p>
      </section>

      <section className="report-section split-list">
        <div>
          <h3>Limitations</h3>
          <ul>
            {scan.report.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Next steps</h3>
          <ul>
            {scan.report.recommended_next_steps.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <p className="disclaimer">{scan.report.disclaimer}</p>
    </section>
  );
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function useFileState() {
  const [current, setCurrent] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!current) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(current);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [current]);

  return { current, previewUrl, setCurrent } as const;
}

function useScanState() {
  const [current, setCurrent] = React.useState<ScanRecord | null>(null);
  return { current, setCurrent } as const;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
