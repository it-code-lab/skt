import React, { useEffect, useMemo, useState } from "react";
import { motion, useAnimationControls } from "framer-motion";

type Path = { d: string };
type Step = { label: string; paths: Path[]; est_ms: number };
type Sketch = { svg: { viewBox: string; strokes: string[] }, steps: Step[] };

type OrderMode = "original" | "top-bottom" | "left-right" | "area";

export default function SketchPlayer({
  data, width, height,
  showGrid = false, gridSize = 50,
  revealColorAtEnd = true, bgSrc,
  playbackSpeed = 1,
  orderMode = "original",
  keepGroups = true,
}: {
  data: Sketch;
  width: number;
  height: number;
  showGrid?: boolean;
  gridSize?: number;
  revealColorAtEnd?: boolean;
  bgSrc?: string;
  playbackSpeed?: number; // 0.5 .. 3
  orderMode?: OrderMode;
  keepGroups?: boolean;   // if true, sort within each step; if false, sort globally
}) {

  // --- helpers to extract geometry from SVG path data "M x y L x y ... Z"
  function metricsForPath(d: string) {
    // Extract all numbers in order; assume alternating x,y for M/L commands (our backend emits like this)
    const nums = Array.from(d.matchAll(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi)).map(m => Number(m[0]));
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i + 1 < nums.length; i += 2) { xs.push(nums[i]); ys.push(nums[i + 1]); }
    if (!xs.length || !ys.length) {
      return { minX: 0, minY: 0, cx: 0, cy: 0, areaBox: 0 };
    }
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
    const areaBox = (maxX - minX) * (maxY - minY); // rough size proxy
    return { minX, minY, cx, cy, areaBox };
  }

  function sorter(mode: OrderMode) {
    return (a: any, b: any) => {
      if (mode === "top-bottom") {
        // smaller Y first (top of canvas)
        if (a.m.minY !== b.m.minY) return a.m.minY - b.m.minY;
        // tie-breaker: left-to-right
        if (a.m.minX !== b.m.minX) return a.m.minX - b.m.minX;
      } else if (mode === "left-right") {
        if (a.m.minX !== b.m.minX) return a.m.minX - b.m.minX;
        if (a.m.minY !== b.m.minY) return a.m.minY - b.m.minY;
      } else if (mode === "area") {
        // big to small (use bounding box area proxy)
        if (a.m.areaBox !== b.m.areaBox) return b.m.areaBox - a.m.areaBox;
        // tie-breaker: top-bottom
        if (a.m.minY !== b.m.minY) return a.m.minY - b.m.minY;
      }
      // default / original (stable by original index)
      return a.idx - b.idx;
    };
  }

  // Build a path-level timeline so *every* path gets animated
  const timeline = useMemo(() => {
    const perStepItems: {
      d: string; step: number; label: string; dur: number; idx: number; m: ReturnType<typeof metricsForPath>
    }[][] = [];

    // capture original index to keep sort stable
    let globalIdx = 0;

    data.steps.forEach((s, stepIndex) => {
      const perPathDur = Math.max(300, Math.round(s.est_ms / Math.max(1, s.paths.length)));
      const arr: any[] = s.paths.map((p, localIdx) => ({
        d: p.d,
        step: stepIndex,
        label: s.label,
        dur: perPathDur,
        idx: globalIdx++,
        m: metricsForPath(p.d),
      }));
      perStepItems.push(arr);
    });

    let orderedFlat: any[];
    if (keepGroups) {
      // sort inside each step, then concatenate in original step order
      const srt = sorter(orderMode);
      const joined = perStepItems.flatMap(arr => [...arr].sort(srt));
      orderedFlat = joined;
    } else {
      // flatten everything and sort across all paths
      const all = perStepItems.flat();
      orderedFlat = [...all].sort(sorter(orderMode));
    }

    // strip metrics from the public timeline objects
    return orderedFlat.map(({ m, ...rest }) => rest) as { d: string; step: number; label: string; dur: number }[];
  }, [data, orderMode, keepGroups]);

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
  React.useEffect(() => {
    let cancelled = false;

    async function run() {
      if (!playing) {
        controls.stop();                 // pause at current progress
        return;
      }
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
      initial={{ strokeDashoffset: "100%" }}
      animate={controls}
      strokeDasharray="100%"
      style={{ willChange: "stroke-dashoffset" }}
    />
  );
}
