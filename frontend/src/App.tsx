import React, { useState } from "react";
import SketchPlayer from "./components/SketchPlayer";

type Sketch = {
  svg: { viewBox: string; strokes: string[] };
  steps: { label: string; paths: { d: string }[]; est_ms: number }[];
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [sketch, setSketch] = useState<Sketch | null>(null);
  const [loading, setLoading] = useState(false);
  const API = import.meta.env.VITE_API_BASE_URL;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/sketch`, { method: "POST", body: fd });
      const data = await res.json();
      setSketch(data);
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "32px auto", padding: 16 }}>
      <h1>Drawing Tutor (MVP)</h1>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, alignItems: "center", margin: "16px 0" }}>
        <input type="file" accept="image/*" onChange={e => setFile(e.target.files?.[0] ?? null)} />
        <button type="submit" disabled={!file || loading}>
          {loading ? "Processing..." : "Detect & Sketch"}
        </button>
      </form>

      {sketch ? <SketchPlayer data={sketch} /> : <p>Upload an image to generate a step-by-step sketch.</p>}
    </div>
  );
}

export default App;
