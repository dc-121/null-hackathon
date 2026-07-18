/**
 * Mount point for one side's crowd.
 *
 * React sets up the canvas and then gets out of the way — the render loop in
 * crowd.ts owns everything inside it, reading `store[side]` directly each
 * frame. Nothing here mirrors emotion into useState.
 */

import { useEffect, useRef } from 'react';
import { startCrowd } from './crowd.js';
import type { Side } from '../state/emotion.js';

interface CrowdPaneProps {
  side: Side;
  label?: string;
}

export function CrowdPane({ side, label }: CrowdPaneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const crowd = startCrowd(canvas, side);
    window.addEventListener('resize', crowd.resize);

    return () => {
      window.removeEventListener('resize', crowd.resize);
      crowd.stop();
    };
  }, [side]);

  return (
    <section className="pane">
      <canvas ref={canvasRef} className="crowd" />
      {label && <span className="label">{label}</span>}
    </section>
  );
}
