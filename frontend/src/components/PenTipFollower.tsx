import React, { useEffect, useRef, useState } from "react";

export function PenTipFollower({ pathRef, progress }:{
  pathRef: React.RefObject<SVGPathElement | null>;
  progress: number; // 0..1 how far along current stroke
}) {
  const [pos, setPos] = useState({ x: 0, y: 0 });
  useEffect(() => {
    const p = pathRef.current;
    if (!p) return;
    const len = p.getTotalLength();
    const pt = p.getPointAtLength(len * progress);
    setPos({ x: pt.x, y: pt.y });
  }, [progress, pathRef]);
  return (
    <g transform={`translate(${pos.x},${pos.y})`}>
      {/* tiny circle as pen tip; replace with a hand PNG if you like */}
      <circle r={4} fill="black" />
    </g>
  );
}
