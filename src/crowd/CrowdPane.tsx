/**
 * Mount point for one side's crowd.
 *
 * React sets up the canvas and then gets out of the way. The render loop owns
 * everything inside it — read `store[side]` directly each frame, never mirror
 * it into useState.
 */

import { useEffect, useRef } from 'react';
import type { Side } from '../state/emotion.js';

interface CrowdPaneProps {
  side: Side;
  label: string;
}

export function CrowdPane({ side, label }: CrowdPaneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
    };
    resize();
    window.addEventListener('resize', resize);

    // TODO: startCrowd(canvas, side) — the figures live here.
    const ctx = canvas.getContext('2d');
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, [side]);

  return (
    <section className="pane">
      <canvas ref={canvasRef} className="crowd" />
      <span className="label">{label}</span>
    </section>
  );
}
