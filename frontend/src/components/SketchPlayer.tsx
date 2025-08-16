import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";

type Path = { d: string };
type Step = { label: string; paths: Path[]; est_ms: number };
type Sketch = { svg: { viewBox: string; strokes: string[] }, steps: Step[] };

export default function SketchPlayer({
  data, width, height,
  showGrid = false, gridSize = 50,
  revealColorAtEnd = true, bgSrc
}: {
  data: Sketch;
  width: number;
  height: number;
  showGrid?: boolean;
  gridSize?: number;
  revealColorAtEnd?: boolean;
  bgSrc?: string;
}) {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [done, setDone] = useState(false);

  const flatPaths = useMemo(() => {
    const out: { d: string; step: number }[] = [];
    data.steps.forEach((s, i) => s.paths.forEach(p => out.push({ d: p.d, step: i })));
    return out;
  }, [data]);

  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= data.steps.length) return;
    const dur = data.steps[stepIdx]?.est_ms ?? 1200;
    const t = setTimeout(() => setStepIdx(i => i + 1), dur);
    return () => clearTimeout(t);
  }, [playing, stepIdx, data.steps]);

  useEffect(() => {
    const finished = stepIdx >= data.steps.length;
    setDone(finished);
  }, [stepIdx, data.steps.length]);

  // For the "currently animating" path, use the same last path from stepIdx (if exists)
  const currentPath = flatPaths[stepIdx];

  // Parse the original viewBox to preserve proportions of uploaded image
  const vb = data.svg.viewBox || `0 0 ${width} ${height}`;

  return (
    <div>
      <svg
        viewBox={vb}
        width={width}
        height={height}
        style={{ background: "#fff", borderRadius: 12, boxShadow: "0 2px 10px rgba(0,0,0,0.08)" }}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Background color image (hidden until finished if revealColorAtEnd) */}
        {bgSrc && (
          <image
            href={bgSrc}
            x="0" y="0"
            width="100%" height="100%"
            opacity={revealColorAtEnd ? (done ? 1 : 0) : 0}
            style={{ transition: "opacity 800ms ease-in-out" }}
            preserveAspectRatio="xMidYMid slice"
          />
        )}

        {/* Optional grid overlay (under the strokes) */}
        {showGrid && (
          <>
            <defs>
              <pattern id="grid" width={gridSize} height={gridSize} patternUnits="userSpaceOnUse">
                <path d={`M ${gridSize} 0 L 0 0 0 ${gridSize}`} fill="none" stroke="rgba(0,0,0,.15)" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect x="0" y="0" width="100%" height="100%" fill="url(#grid)" />
          </>
        )}

        {/* Already-drawn strokes */}
        {flatPaths.slice(0, stepIdx).map((p, i) => (
          <path key={`drawn-${i}`} d={p.d} fill="none" stroke="black" strokeWidth={2}/>
        ))}

        {/* Currently animating stroke */}
        {currentPath && <AnimatedStroke d={currentPath.d} />}

      </svg>

      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button onClick={() => setPlaying(p => !p)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setPlaying(false); setStepIdx(i => Math.max(0, i - 1)); }}>Back</button>
        <button onClick={() => { setPlaying(false); setStepIdx(i => Math.min(data.steps.length, i + 1)); }}>Next</button>
        <button onClick={() => { setPlaying(false); setStepIdx(0); }}>Reset</button>
      </div>

      <div style={{ marginTop: 8, opacity: 0.8 }}>
        {stepIdx < data.steps.length ? data.steps[stepIdx]?.label : "Done"}
      </div>
    </div>
  );
}

function AnimatedStroke({ d }: { d: string }) {
  const ref = useRef<SVGPathElement>(null);
  const [len, setLen] = useState(0);
  useEffect(() => { if (ref.current) setLen(ref.current.getTotalLength()); }, [d]);
  return (
    <motion.path
      ref={ref}
      d={d}
      fill="none"
      stroke="black"
      strokeWidth={2}
      strokeDasharray={len}
      initial={{ strokeDashoffset: len }}
      animate={{ strokeDashoffset: 0 }}
      transition={{ duration: Math.max(0.6, Math.min(6, len / 300)), ease: "easeInOut" }}
    />
  );
}
