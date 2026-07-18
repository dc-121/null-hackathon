/**
 * The crowd.
 *
 * Emotion is represented by figures, and the one rule that matters is that
 * they are ANONYMOUS AND UNDIFFERENTIATED. Same figure, different behaviour.
 * The moment one becomes "the angry one" we've smuggled discrete emotion
 * categories back in — the exact thing we rejected — and it starts looking
 * like Inside Out.
 *
 * So: feeling lives in how they MOVE and CLUSTER, never in which figure it is.
 *
 * Why a crowd at all: it's the only representation that holds ambivalence
 * honestly. Two feelings at once don't average to neutral the way a point in
 * a coordinate space does — both populations are simply present, in tension.
 *
 * Mapping:
 *   intensity -> how many, how fast, how jittery
 *   effort    -> cohesion: pulled tight together vs scattered
 *   movement  -> wander, how much they turn
 *   emphasis  -> an impulse that ripples outward from a point
 */

import { store, trimEmphases, type Side } from '../state/emotion.js';

const MAX_AGENTS = 240;
const BASE_AGENTS = 30;

interface Agent {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Gait phase, so legs don't all move in lockstep. */
  phase: number;
  /** Per-agent variation so the crowd doesn't look mechanical. */
  bias: number;
}

function makeAgent(w: number, h: number): Agent {
  return {
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 2,
    vy: (Math.random() - 0.5) * 2,
    phase: Math.random() * Math.PI * 2,
    bias: 0.7 + Math.random() * 0.6,
  };
}

/** Draw one anonymous figure: head, body, two legs, arms. Mid-stride. */
function drawFigure(
  ctx: CanvasRenderingContext2D,
  a: Agent,
  scale: number,
  alpha: number
): void {
  const s = scale;
  const swing = Math.sin(a.phase) * 0.9;
  const bob = Math.abs(Math.cos(a.phase)) * s * 0.12;

  ctx.save();
  ctx.translate(a.x, a.y - bob);
  // Lean into the direction of travel — reads as running rather than sliding.
  ctx.rotate(Math.atan2(a.vy, a.vx) * 0.06);
  ctx.globalAlpha = alpha;
  ctx.lineWidth = Math.max(1, s * 0.16);
  ctx.lineCap = 'round';

  // body
  ctx.beginPath();
  ctx.moveTo(0, -s * 0.35);
  ctx.lineTo(0, s * 0.3);
  ctx.stroke();

  // legs
  ctx.beginPath();
  ctx.moveTo(0, s * 0.3);
  ctx.lineTo(swing * s * 0.5, s);
  ctx.moveTo(0, s * 0.3);
  ctx.lineTo(-swing * s * 0.5, s);
  ctx.stroke();

  // arms, counter-swung
  ctx.beginPath();
  ctx.moveTo(0, -s * 0.15);
  ctx.lineTo(-swing * s * 0.42, s * 0.15);
  ctx.moveTo(0, -s * 0.15);
  ctx.lineTo(swing * s * 0.42, s * 0.15);
  ctx.stroke();

  // head
  ctx.beginPath();
  ctx.arc(0, -s * 0.6, s * 0.26, 0, Math.PI * 2);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();

  ctx.restore();
}

export interface CrowdHandle {
  stop(): void;
}

export function startCrowd(canvas: HTMLCanvasElement, side: Side): CrowdHandle {
  const ctx = canvas.getContext('2d');
  if (!ctx) return { stop() {} };

  const agents: Agent[] = [];
  /** Emphasis events already turned into an impulse, so we don't re-fire. */
  const consumed = new Set<number>();
  let raf = 0;
  let stopped = false;

  const frame = () => {
    if (stopped) return;
    raf = requestAnimationFrame(frame);

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;
    if (!w || !h) return;

    const { affect, emphases } = store[side];
    trimEmphases();

    // Normalise the z-scored channels into 0..1 drive values.
    const intensity = Math.max(0, Math.min(1, (affect.intensity + 1) / 3));
    const effort = Math.max(0, Math.min(1, (affect.effort + 1) / 3));
    const movement = Math.max(0, Math.min(1, affect.movement / 2));

    // Quantity is itself an encoding: feeling more means more of them.
    const want = Math.round(BASE_AGENTS + intensity * (MAX_AGENTS - BASE_AGENTS));
    while (agents.length < want) agents.push(makeAgent(w, h));
    if (agents.length > want) agents.length = want;

    // Cohesion stays a mild ATTRACTION always — it's what keeps the crowd on
    // screen. Dispersal comes from local repulsion instead (below): inverted
    // gravity scales with distance, so it runs away and piles everyone on the
    // wrap edges.
    //
    // Effort then sets the balance: pulled tight together vs given room.
    const cx = w / 2;
    const cy = h / 2;
    const cohesion = 0.0002 + effort * 0.0011;
    const personalSpace = (46 - effort * 30) * dpr;
    const speed = 0.4 + intensity * 3.2;
    const jitter = 0.05 + intensity * 0.55;
    const turn = 0.02 + movement * 0.22;

    // New emphasis events become an outward impulse — a syllable you leaned
    // on visibly disturbs the room.
    for (const e of emphases) {
      if (consumed.has(e.id)) continue;
      consumed.add(e.id);
      const ox = Math.random() * w;
      const oy = Math.random() * h;
      for (const a of agents) {
        const dx = a.x - ox;
        const dy = a.y - oy;
        const dist = Math.hypot(dx, dy) || 1;
        if (dist > Math.max(w, h) * 0.45) continue;
        const push = (e.weight * 14 * dpr) / dist;
        a.vx += (dx / dist) * push;
        a.vy += (dy / dist) * push;
      }
    }
    if (consumed.size > 512) consumed.clear();

    for (const a of agents) {
      // wander
      const angle = Math.atan2(a.vy, a.vx) + (Math.random() - 0.5) * turn;
      const mag = Math.hypot(a.vx, a.vy) || 0.001;
      a.vx = Math.cos(angle) * mag;
      a.vy = Math.sin(angle) * mag;

      a.vx += (Math.random() - 0.5) * jitter;
      a.vy += (Math.random() - 0.5) * jitter;

      a.vx += (cx - a.x) * cohesion;
      a.vy += (cy - a.y) * cohesion;

      // Personal space. Without it they stack into an illegible pile in the
      // middle. Sampled rather than all-pairs — visually identical, and keeps
      // this O(n) instead of O(n^2).
      for (let s = 0; s < 6; s++) {
        const other = agents[(Math.random() * agents.length) | 0];
        if (other === a) continue;
        const dx = a.x - other.x;
        const dy = a.y - other.y;
        const d2 = dx * dx + dy * dy;
        const near = personalSpace;
        if (d2 > 0.01 && d2 < near * near) {
          const d = Math.sqrt(d2);
          const push = ((near - d) / near) * 0.5 * dpr;
          a.vx += (dx / d) * push;
          a.vy += (dy / d) * push;
        }
      }

      // Damp toward the speed this affect implies, rather than clamping —
      // acceleration and settling both read as feeling.
      const target = speed * a.bias * dpr;
      const current = Math.hypot(a.vx, a.vy) || 0.001;
      const adjust = 1 + (target / current - 1) * 0.08;
      a.vx *= adjust;
      a.vy *= adjust;

      a.x += a.vx;
      a.y += a.vy;

      // wrap
      if (a.x < -20) a.x = w + 20;
      if (a.x > w + 20) a.x = -20;
      if (a.y < -20) a.y = h + 20;
      if (a.y > h + 20) a.y = -20;

      a.phase += 0.08 + current * 0.035;
    }

    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#e8e6e3';
    const scale = 9 * dpr;
    const alpha = 0.35 + intensity * 0.5;
    for (const a of agents) drawFigure(ctx, a, scale, alpha);
  };

  raf = requestAnimationFrame(frame);

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
    },
  };
}
