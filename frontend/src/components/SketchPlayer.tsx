import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";

type Path = { d: string };
type Step = { label: string; paths: Path[]; est_ms: number };
type Sketch = { svg: { viewBox: string; strokes: string[] }, steps: Step[] };

export default function SketchPlayer({ data }: { data: Sketch }) {
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const flatPaths = useMemo(() => {
    const out: { d: string; step: number }[] = [];
    data.steps.forEach((s, i) => s.paths.forEach(p => out.push({ d: p.d, step: i })));
    return out;
  }, [data]);

  useEffect(() => {
    if (!playing) return;
    const dur = data.steps[stepIdx]?.est_ms ?? 1200;
    const t = setTimeout(() => setStepIdx(i => Math.min(i + 1, data.steps.length)), dur);
    return () => clearTimeout(t);
  }, [playing, stepIdx, data.steps]);

  const currentPath = flatPaths[stepIdx];

  return (
    <div className="w-full">
      <svg viewBox={data.svg.viewBox} style={{ width: "100%", background: "#fff" }}>
        {flatPaths.slice(0, stepIdx).map((p, i) => (
          <path key={`drawn-${i}`} d={p.d} fill="none" stroke="black" strokeWidth={2}/>
        ))}
        {currentPath && <AnimatedStroke d={currentPath.d} />}
      </svg>

      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button onClick={() => setPlaying(p => !p)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => setStepIdx(i => Math.max(0, i-1))}>Back</button>
        <button onClick={() => setStepIdx(i => Math.min(data.steps.length, i+1))}>Next</button>
        <button onClick={() => { setStepIdx(0); setPlaying(false); }}>Reset</button>
      </div>

      <div style={{ marginTop: 8, opacity: 0.8 }}>
        {data.steps[stepIdx]?.label ?? "Done"}
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
      transition={{ duration: Math.max(0.6, Math.min(6, len/300)), ease: "easeInOut" }}
    />
  );
}
