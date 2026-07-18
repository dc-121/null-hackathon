/**
 * Gaits.
 *
 * How a body carries itself is most of what makes an emotion legible — long
 * before colour or silhouette register. A skip and a trudge are unmistakable
 * from across a room, at any resolution.
 *
 * These are built from the anatomy of a real walk cycle rather than a sine
 * wave, because that's where the character lives:
 *
 *  - The body dips TWICE per stride (once per footfall) and rises at mid-
 *    stance. That's `bounce` at 2x phase. Getting this wrong is why naive
 *    walk cycles look like floating.
 *  - Where the accent sits is the whole personality. Happy accents the RISE
 *    (airtime, hang). Angry accents the DROP (stomp, weight going down hard).
 *    Sad has almost no vertical at all — the weight never leaves the floor.
 *  - Stance/swing asymmetry: sad drags (long stance), happy floats (long
 *    swing). A pure sine is symmetric and therefore characterless.
 *  - Arms are the giveaway. Big counter-swing with the elbows out reads as
 *    open and confident; a dead, trailing arm reads as defeated; a tight,
 *    high, tense swing reads as aggressive.
 */

import type { Gait } from './archetypes.js';

export interface Pose {
  /** Leg swing, radians. Right leg is the negation. */
  leg: number;
  /** Arm swing, radians. Counter-swung against the legs. */
  arm: number;
  /** Elbows out from the body, radians. Openness vs tightness. */
  armSpread: number;
  /** Vertical offset, world units before height scaling. */
  bounce: number;
  /** Extra forward lean this gait implies. */
  lean: number;
  /** Side-to-side sway, radians. */
  sway: number;
  /** Head pitch on top of posture. Positive = chin down. */
  headDrop: number;
}

/** Skewed oscillator: `k` > 0 lengthens the rise, < 0 lengthens the fall. */
function skew(phase: number, k: number): number {
  return Math.sin(phase + k * Math.sin(phase));
}

export function pose(gait: Gait, phase: number, energy: number, seed: number): Pose {
  const s = Math.sin(phase);
  const c = Math.cos(phase);
  /** Footfall oscillator — twice per stride, as in a real cycle. */
  const step = Math.abs(Math.cos(phase));

  switch (gait) {
    case 'skip': {
      // Happy. Long airtime, hang at the top, light landing. Arms high and
      // open, chest lifted. The lift is on the OFF-beat of the footfall,
      // which is what makes a skip feel like a skip rather than a bounce.
      const air = Math.abs(Math.sin(phase * 0.5)) ** 0.55;
      return {
        leg: skew(phase, 0.35) * (0.75 + energy * 0.5) + 0.22,
        arm: -skew(phase, 0.35) * (0.8 + energy * 0.6) - 0.3,
        armSpread: 0.3 + energy * 0.15,
        bounce: air * (0.2 + energy * 0.26),
        lean: -0.1,
        sway: c * 0.09,
        headDrop: -0.12,
      };
    }

    case 'stomp': {
      // Angry. The accent is DOWNWARD — weight driven into the floor on each
      // footfall, then a hard recovery. Wide stance, swagger, arms tight and
      // high, shoulders up.
      const impact = step ** 3;
      return {
        leg: skew(phase, -0.3) * (0.7 + energy * 0.5),
        arm: -skew(phase, -0.3) * (0.7 + energy * 0.4),
        armSpread: 0.42 + energy * 0.2,
        bounce: (1 - impact) * (0.07 + energy * 0.1),
        lean: 0.26 + energy * 0.14,
        sway: s * 0.17,
        headDrop: 0.06,
      };
    }

    case 'trudge': {
      // Sad. Feet barely clear the floor, long dragging stance, weight sinking
      // rather than lifting. Arms dead and trailing slightly behind. This is
      // mostly an exercise in taking things AWAY.
      return {
        leg: skew(phase, -0.5) * 0.26,
        arm: -skew(phase, -0.5) * 0.1 - 0.12,
        armSpread: -0.06,
        bounce: -step * 0.02,
        lean: 0.12,
        sway: s * 0.06,
        headDrop: 0.34,
      };
    }

    case 'scurry': {
      // Afraid. Short, fast, hesitant steps. Crouched, arms tucked in tight
      // across the body, head low and darting.
      const dart = Math.sin(phase * 0.33 + seed);
      return {
        leg: s * (0.4 + energy * 0.3),
        arm: -s * 0.28,
        armSpread: -0.3,
        bounce: step * (0.02 + energy * 0.04),
        lean: 0.2,
        sway: dart * 0.14,
        headDrop: 0.2 + dart * 0.12,
      };
    }

    case 'run': {
      // Full flight phase — both feet off the ground at the extremes.
      return {
        leg: s * (0.95 + energy * 0.6),
        arm: -s * (0.85 + energy * 0.5),
        armSpread: 0.16,
        bounce: (1 - step) * (0.09 + energy * 0.14),
        lean: 0.24 + energy * 0.18,
        sway: 0,
        headDrop: 0.04,
      };
    }

    case 'creep': {
      // Guilty. Rolling, careful steps that avoid making a sound. Shoulders
      // pulled in, head down, arms held close.
      return {
        leg: skew(phase, -0.4) * 0.32,
        arm: -s * 0.12,
        armSpread: -0.22,
        bounce: step * 0.012,
        lean: 0.18,
        sway: 0,
        headDrop: 0.3,
      };
    }

    case 'stagger': {
      // Surprised. Off-balance, catching itself. The seed decorrelates agents
      // so a crowd of these doesn't lurch in unison.
      const wobble = Math.sin(phase * 0.55 + seed) * 0.4;
      return {
        leg: s * (0.6 + energy * 0.5) + wobble * 0.3,
        arm: -s * (0.5 + energy * 0.4) + wobble * 0.5,
        armSpread: 0.34 + Math.abs(wobble) * 0.3,
        bounce: step * (0.05 + energy * 0.07),
        lean: 0.1 + wobble * 0.12,
        sway: wobble * 0.16,
        headDrop: -0.14,
      };
    }

    case 'drift': {
      // Loving. Unhurried, floating, faintly swaying. Barely a gait at all.
      return {
        leg: s * 0.22,
        arm: -s * 0.2,
        armSpread: 0.12,
        bounce: Math.sin(phase * 0.6) * 0.04,
        lean: -0.05,
        sway: Math.sin(phase * 0.4 + seed) * 0.11,
        headDrop: -0.06,
      };
    }

    case 'walk':
    default: {
      // Calm. The neutral reference the others are heard against.
      return {
        leg: s * (0.5 + energy * 0.5),
        arm: -s * (0.42 + energy * 0.45),
        armSpread: 0.1,
        bounce: (1 - step) * (0.035 + energy * 0.07),
        lean: 0,
        sway: s * 0.04,
        headDrop: 0,
      };
    }
  }
}
