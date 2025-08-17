import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, useAnimationControls } from "framer-motion";

type Path = { d: string };
type Step = { label: string; paths: Path[]; est_ms: number };
type Sketch = { svg: { viewBox: string; strokes: string[] }, steps: Step[] };

export default function SketchPlayer({
  data, width, height,
  showGrid = false, gridSize = 50,
  revealColorAtEnd = true, bgSrc,
  playbackSpeed = 1
}: {
  data: Sketch;
  width: number;
  height: number;
  showGrid?: boolean;
  gridSize?: number;
  revealColorAtEnd?: boolean;
  bgSrc?: string;
  playbackSpeed?: number; // 0.5 .. 3
}) {
  // Build a path-level timeline so *every* path gets animated
  const timeline = useMemo(() => {
    const items: { d: string; step: number; label: string; dur: number }[] = [];
    data.steps.forEach((s, stepIndex) => {
      const perPath = Math.max(300, Math.round(s.est_ms / Math.max(1, s.paths.length)));
      s.paths.forEach(p => items.push({ d: p.d, step: stepIndex, label: s.label, dur: perPath }));
    });
    return items;
  }, [data]);

  const [pathIdx, setPathIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const vb = data.svg.viewBox || `0 0 ${width} ${height}`;
  const done = pathIdx >= timeline.length;
  const current = !done ? timeline[pathIdx] : null;

  return (
    <div>
      <svg
        viewBox={vb}
        width={width}
        height={height}
        style={{ background: "#fff", borderRadius: 12, boxShadow: "0 2px 10px rgba(0,0,0,0.08)" }}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Background image (revealed at the end if enabled) */}
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

        {/* Optional grid */}
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

        {/* Already drawn strokes */}
        {timeline.slice(0, pathIdx).map((p, i) => (
          <path
            key={`drawn-${i}`}
            d={p.d}
            fill="none"
            stroke="black"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {/* Current stroke animates as a "write-on" */}
        {!done && current && (
          <AnimatedStroke
            key={pathIdx}                 // force fresh animation when index changes
            d={current.d}
            durationMs={Math.max(120, current.dur / Math.max(0.1, playbackSpeed))}
            playing={playing}
            onDone={() => setPathIdx(i => i + 1)}
          />
        )}
      </svg>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={() => setPlaying(p => !p)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setPlaying(false); setPathIdx(i => Math.max(0, i - 1)); }}>Back</button>
        <button onClick={() => { setPlaying(false); setPathIdx(i => Math.min(timeline.length, i + 1)); }}>Next</button>
        <button onClick={() => { setPlaying(false); setPathIdx(0); }}>Reset</button>

        <label style={{ marginLeft: 12 }}>
          Speed:&nbsp;
          <span style={{ fontVariantNumeric: "tabular-nums" }}>{playbackSpeed.toFixed(1)}×</span>
        </label>
      </div>

      <div style={{ marginTop: 8, opacity: 0.8 }}>
        {!done ? data.steps[timeline[pathIdx].step]?.label : "Done"}
      </div>
    </div>
  );
}

/** Draws a path by animating strokeDashoffset. Pauses/resumes cleanly. */
function AnimatedStroke({
  d,
  durationMs,
  stroke = "black",
  strokeWidth = 2,
  playing = true,
  onDone,
}: {
  d: string;
  durationMs: number;
  stroke?: string;
  strokeWidth?: number;
  playing?: boolean;
  onDone?: () => void;
}) {
  const controls = useAnimationControls();

  // Start / pause / resume the draw animation
  useEffect(() => {
    let cancelled = false;

    async function run() {
      if (!playing) {
        controls.stop();                 // pause at current progress
        return;
      }
      // Animate strokeDashoffset from the path's length down to 0
      await controls.start({
        strokeDashoffset: 0,
        transition: {
          duration: Math.max(0.1, durationMs / 1000),
          ease: "easeInOut",
        },
      });
      if (!cancelled) onDone?.();
    }

    run();
    return () => { cancelled = true; };
  }, [playing, durationMs, controls, onDone, d]);

  return (
    <motion.path
      d={d}
      fill="none"
      stroke={stroke}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      vectorEffect="non-scaling-stroke"
      // Use stroke-dashoffset for the "draw" effect
      initial={{ strokeDashoffset: "100%" }}
      animate={controls}
      // Set strokeDasharray to make the full path visible and animatable
      strokeDasharray="100%"
      style={{ willChange: "stroke-dashoffset" }}
    />
  );
}
