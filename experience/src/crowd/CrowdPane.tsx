/**
 * Mount point for one side's crowd.
 *
 * React sets up the canvas and then gets out of the way — the render loop in
 * crowd.ts owns everything inside it, reading `store[side]` directly each
 * frame. Nothing here mirrors emotion into useState.
 */

import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react';
import { startCrowd } from './crowd.js';
import type { Emotion, Side } from '../state/emotion.js';

interface CrowdPaneProps {
  side: Side;
  label: string;
  title: string;
  emotion: Emotion | null;
  confidence: number;
  metricLabel?: string;
  children?: ReactNode;
}

export function CrowdPane({
  side,
  label,
  title,
  emotion,
  confidence,
  metricLabel = 'fused confidence',
  children,
}: CrowdPaneProps) {
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
    <section
      className={`pane pane--${side}`}
      data-emotion={emotion ?? 'waiting'}
      style={{
        '--pane-emotion-mix': `${Math.round(10 + Math.max(0, Math.min(1, confidence)) * 38)}%`,
      } as CSSProperties}
      aria-label={`${title} emotion crowd`}
    >
      <canvas ref={canvasRef} className="crowd" />
      <div className="pane-vignette" aria-hidden="true" />
      <header className="pane-heading">
        <span className="pane-kicker">{label}</span>
        <div>
          <strong>{emotion ?? 'waiting for signal'}</strong>
          <span>{emotion ? `${Math.round(confidence * 100)}% ${metricLabel}` : title}</span>
        </div>
      </header>
      <div className="pane-content">{children}</div>
    </section>
  );
}
