/**
 * Gaits.
 *
 * How a body carries itself is most of what makes an emotion legible at a
 * glance — long before colour or silhouette register. A skip and a trudge are
 * unmistakable from across a room.
 *
 * Each returns the pose for one frame. `phase` advances with speed, so a gait
 * automatically quickens as the agent does.
 */

import type { Gait } from './archetypes.js';

export interface Pose {
  /** Leg swing, radians. Right leg is the negation. */
  leg: number;
  /** Arm swing, radians. Counter-swung against the legs. */
  arm: number;
  /** Vertical offset, world units before height scaling. */
  bounce: number;
  /** Extra forward lean this gait implies. */
  lean: number;
  /** Side-to-side sway, radians. */
  sway: number;
}

export function pose(gait: Gait, phase: number, energy: number, seed: number): Pose {
  const s = Math.sin(phase);
  const c = Math.cos(phase);

  switch (gait) {
    case 'skip':
      // Asymmetric double-hop: one leg leads, and the body is airborne for
      // most of the cycle. Reads as delight without any other cue.
      return {
        leg: s * (0.7 + energy * 0.5) + 0.25,
        arm: -s * (0.6 + energy * 0.6) - 0.35,
        bounce: Math.abs(Math.sin(phase * 0.5)) ** 0.6 * (0.16 + energy * 0.22),
        lean: -0.08,
        sway: c * 0.08,
      };

    case 'run':
      return {
        leg: s * (0.95 + energy * 0.6),
        arm: -s * (0.85 + energy * 0.5),
        bounce: Math.abs(c) * (0.06 + energy * 0.12),
        lean: 0.22 + energy * 0.18,
        sway: 0,
      };

    case 'trudge':
      // Barely lifts its feet. Weight forward, arms nearly still.
      return {
        leg: s * 0.28,
        arm: -s * 0.14,
        bounce: Math.abs(c) * 0.018,
        lean: 0.1,
        sway: s * 0.05,
      };

    case 'creep':
      return {
        leg: s * 0.34,
        arm: -s * 0.1,
        bounce: Math.abs(c) * 0.012,
        lean: 0.16,
        sway: 0,
      };

    case 'stagger': {
      // Irregular on purpose — the seed decorrelates agents so a crowd of
      // these doesn't pulse in unison.
      const wobble = Math.sin(phase * 0.55 + seed) * 0.4;
      return {
        leg: s * (0.6 + energy * 0.5) + wobble * 0.3,
        arm: -s * (0.5 + energy * 0.4) + wobble * 0.5,
        bounce: Math.abs(c) * (0.04 + energy * 0.08),
        lean: 0.12 + wobble * 0.12,
        sway: wobble * 0.16,
      };
    }

    case 'drift':
      return {
        leg: s * 0.2,
        arm: -s * 0.16,
        bounce: Math.sin(phase * 0.6) * 0.035,
        lean: -0.04,
        sway: Math.sin(phase * 0.4 + seed) * 0.1,
      };

    case 'walk':
    default:
      return {
        leg: s * (0.5 + energy * 0.5),
        arm: -s * (0.42 + energy * 0.45),
        bounce: Math.abs(c) * (0.03 + energy * 0.07),
        lean: 0,
        sway: 0,
      };
  }
}
