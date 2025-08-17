import React, { useMemo, useState } from "react";
import SketchPlayer from "./components/SketchPlayer";

type Sketch = {
  svg: { viewBox: string; strokes: string[] };
  steps: { label: string; paths: { d: string }[]; est_ms: number }[];
};

const ASPECTS = {
  "16:9": 16 / 9,
  "9:16": 9 / 16,
  "1:1": 1,
} as const;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [sketch, setSketch] = useState<Sketch | null>(null);
  const [loading, setLoading] = useState(false);

  const [aspectKey, setAspectKey] = useState<keyof typeof ASPECTS>("16:9");
  const [targetWidth, setTargetWidth] = useState<number>(1920);
  const aspect = ASPECTS[aspectKey];
  const targetHeight = useMemo(() => Math.round(targetWidth / aspect), [targetWidth, aspect]);

  const [showGrid, setShowGrid] = useState(true);
  const [gridSize, setGridSize] = useState(50);
  const [revealColorAtEnd, setRevealColorAtEnd] = useState(true);

  const [mode, setMode] = useState<"auto"|"cartoon"|"photo">("auto");
  const [detail, setDetail] = useState(10);
  const [vector, setVector] = useState<"outline"|"centerline">("centerline");

  const [playbackSpeed, setPlaybackSpeed] = useState(0.5); // 0.5 .. 3

  // NEW: ordering + grouping options
  const [orderMode, setOrderMode] = useState<"original" | "top-bottom" | "left-right" | "area">("top-bottom");
  const [keepGroups, setKeepGroups] = useState<boolean>(false);

  const API = import.meta.env.VITE_API_BASE_URL;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/sketch?mode=${mode}&detail=${detail}&vector=${vector}`, { method: "POST", body: fd });
      const data = await res.json();
      setSketch(data);
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      setFileUrl(URL.createObjectURL(file));
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: "32px auto", padding: 16 }}>
      {/* <h1>Drawing Tutor (MVP)</h1> */}

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, alignItems: "center", margin: "16px 0", flexWrap: "wrap" }}>
        <input
          type="file"
          accept="image/*"
          onChange={e => {
            const f = e.target.files?.[0] ?? null;
            setFile(f);
          }}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? "Processing..." : "Detect & Sketch"}
        </button>

        <div style={{ display: "flex", gap: 8, alignItems: "center", marginLeft: 16 }}>
          <label>Aspect:</label>
          {Object.keys(ASPECTS).map(k => (
            <button
              key={k}
              type="button"
              onClick={() => setAspectKey(k as any)}
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: k === aspectKey ? "2px solid #111" : "1px solid #ccc",
                background: k === aspectKey ? "#f0f0f0" : "white",
              }}
            >
              {k}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label>Width(px):</label>
          <input
            type="number"
            min={720}
            step={10}
            value={targetWidth}
            onChange={e => setTargetWidth(parseInt(e.target.value || "0", 10))}
            style={{ width: 100 }}
          />
          <span>→ Height: {targetHeight}px</span>
        </div>

        <label style={{ marginLeft: 12 }}>
          <input type="checkbox" checked={showGrid} onChange={e => setShowGrid(e.target.checked)} /> Grid
        </label>
        {showGrid && (
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <label>Grid size:</label>
            <input type="number" min={10} max={400} step={5} value={gridSize} onChange={e => setGridSize(parseInt(e.target.value || "50", 10))} style={{ width: 80 }} />
          </div>
        )}

        <label style={{ marginLeft: 12 }}>
          <input type="checkbox" checked={revealColorAtEnd} onChange={e => setRevealColorAtEnd(e.target.checked)} /> Reveal original at end
        </label>

        <select value={mode} onChange={e=>setMode(e.target.value as any)}>
          <option value="auto">Auto</option>
          <option value="cartoon">Cartoon / Line Art</option>
          <option value="photo">Photo / Illustration</option>
        </select>
        <label>Detail: {detail}
          <input type="range" min={1} max={10} value={detail}
                onChange={e=>setDetail(parseInt(e.target.value,10))}/>
        </label>

        <select value={vector} onChange={e=>setVector(e.target.value as any)}>
          <option value="centerline">Centerline (strokes)</option>
          <option value="outline">Outline (regions)</option>
        </select>

        {/* Playback speed */}
        <label style={{ marginLeft: 12 }}>Speed:
          <input
            type="range"
            min={0.5} max={3} step={0.1}
            value={playbackSpeed}
            onChange={e => setPlaybackSpeed(parseFloat(e.target.value))}
            style={{ width: 140, verticalAlign: "middle", marginLeft: 6 }}
          />
          <span style={{ marginLeft: 6 }}>{playbackSpeed.toFixed(1)}×</span>
        </label>

        {/* NEW: order + grouping controls */}
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginLeft: 12 }}>
          <label>Order:</label>
          <select value={orderMode} onChange={e => setOrderMode(e.target.value as any)}>
            <option value="original">Original</option>
            <option value="top-bottom">Top → Bottom</option>
            <option value="left-right">Left → Right</option>
            <option value="area">Area (big → small)</option>
          </select>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={keepGroups} onChange={e => setKeepGroups(e.target.checked)} />
            Keep step groups
          </label>
        </div>
      </form>

      {sketch ? (
        <SketchPlayer
          data={sketch}
          width={targetWidth}
          height={targetHeight}
          showGrid={showGrid}
          gridSize={gridSize}
          revealColorAtEnd={revealColorAtEnd}
          bgSrc={fileUrl || undefined}
          playbackSpeed={playbackSpeed}
          orderMode={orderMode}
          keepGroups={keepGroups}
        />
      ) : (
        <p>Upload an image to generate a step-by-step sketch.</p>
      )}
    </div>
  );
}

export default App;
