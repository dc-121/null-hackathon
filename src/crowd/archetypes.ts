/**
 * The nine.
 *
 * `coefficient` is the loading each emotion carries on the primary axis — it
 * sets the RESTING mix when nothing much is happening. `affinity` then shifts
 * that mix continuously with affect.
 *
 * Note what this deliberately isn't: affect never picks a winner. There's no
 * threshold and no argmax anywhere. What you read is the shifting population —
 * so two feelings at once show up as two populations on screen, which is the
 * one thing a single label can never express.
 */

export type Gait = 'walk' | 'skip' | 'run' | 'trudge' | 'creep' | 'stagger' | 'drift';
export type Behavior =
  | 'wander'
  | 'strike' // seeks others out and swings at them
  | 'flee' // avoids everyone, bolts from the cursor
  | 'frantic' // runs hard, changes direction constantly, sweats
  | 'approach' // gravitates toward others and lingers
  | 'withdraw' // drifts to the edges, keeps its distance
  | 'startle'; // freezes, then recoils

export interface Archetype {
  id: string;
  /** Loading on the primary axis. Sets the resting proportion. */
  coefficient: number;

  // palette
  hue: number;
  sat: number;
  light: number;
  hairHue: number;
  hairSat: number;
  hairLight: number;

  // silhouette
  headScale: number;
  girth: number;
  limbLength: number;
  heightBias: number;
  /** 0 = cropped, 1 = tall spiked crest. */
  hair: number;

  // posture, layered on top of the crowd-wide posture
  leanBias: number;
  hunchBias: number;
  bounceBias: number;
  speedBias: number;

  gait: Gait;
  behavior: Behavior;

  /** Multiplier on the resting proportion, given current affect. */
  affinity(intensity: number, effort: number, movement: number): number;
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

export const ARCHETYPES: Archetype[] = [
  {
    id: 'calm',
    coefficient: 0.2206,
    hue: 0.52, sat: 0.32, light: 0.62,
    hairHue: 0.55, hairSat: 0.25, hairLight: 0.4,
    headScale: 1.0, girth: 1.05, limbLength: 1.0, heightBias: 1.02, hair: 0.25,
    leanBias: -0.05, hunchBias: -0.15, bounceBias: 0.5, speedBias: 0.6,
    gait: 'walk', behavior: 'wander',
    affinity: (i, e, m) => clamp01(1 - i) * clamp01(1 - e) * clamp01(1 - m * 0.6) * 2.4,
  },
  {
    id: 'desperate',
    coefficient: 0.0347,
    hue: 0.09, sat: 0.6, light: 0.5,
    hairHue: 0.05, hairSat: 0.5, hairLight: 0.25,
    headScale: 1.02, girth: 0.92, limbLength: 1.04, heightBias: 0.99, hair: 0.7,
    leanBias: 0.34, hunchBias: 0.3, bounceBias: 1.3, speedBias: 1.9,
    gait: 'run', behavior: 'frantic',
    affinity: (i, e, m) => i * e * clamp01(0.4 + m) * 3.2,
  },
  {
    id: 'surprised',
    coefficient: 0.0338,
    hue: 0.15, sat: 0.72, light: 0.66,
    hairHue: 0.12, hairSat: 0.55, hairLight: 0.45,
    headScale: 1.24, girth: 0.95, limbLength: 0.98, heightBias: 1.0, hair: 0.85,
    leanBias: -0.22, hunchBias: -0.1, bounceBias: 1.5, speedBias: 1.1,
    gait: 'stagger', behavior: 'startle',
    affinity: (i, _e, m) => i * m * 3.4,
  },
  {
    id: 'angry',
    coefficient: 0.0291,
    hue: 0.99, sat: 0.75, light: 0.46,
    hairHue: 0.02, hairSat: 0.6, hairLight: 0.22,
    headScale: 0.94, girth: 1.3, limbLength: 0.94, heightBias: 1.03, hair: 0.9,
    leanBias: 0.42, hunchBias: 0.4, bounceBias: 0.8, speedBias: 1.45,
    gait: 'stagger', behavior: 'strike',
    affinity: (i, e) => i * e * e * 3.6,
  },
  {
    id: 'happy',
    coefficient: -0.0419,
    hue: 0.14, sat: 0.85, light: 0.62,
    hairHue: 0.1, hairSat: 0.7, hairLight: 0.5,
    headScale: 1.14, girth: 0.98, limbLength: 1.0, heightBias: 1.0, hair: 0.6,
    leanBias: -0.16, hunchBias: -0.35, bounceBias: 2.4, speedBias: 1.2,
    gait: 'skip', behavior: 'approach',
    affinity: (i, e) => i * clamp01(1 - e) * 3.4,
  },
  {
    id: 'sad',
    coefficient: -0.0631,
    hue: 0.6, sat: 0.4, light: 0.42,
    hairHue: 0.62, hairSat: 0.3, hairLight: 0.24,
    headScale: 1.02, girth: 0.88, limbLength: 1.12, heightBias: 0.95, hair: 0.4,
    leanBias: 0.24, hunchBias: 0.75, bounceBias: 0.2, speedBias: 0.42,
    gait: 'trudge', behavior: 'withdraw',
    affinity: (i, _e, m) => clamp01(1 - i) * clamp01(1 - m) * 2.2,
  },
  {
    id: 'loving',
    coefficient: -0.0665,
    hue: 0.94, sat: 0.5, light: 0.66,
    hairHue: 0.9, hairSat: 0.35, hairLight: 0.4,
    headScale: 1.08, girth: 1.0, limbLength: 1.0, heightBias: 1.0, hair: 0.55,
    leanBias: -0.08, hunchBias: -0.2, bounceBias: 0.9, speedBias: 0.7,
    gait: 'drift', behavior: 'approach',
    affinity: (i, e, m) => clamp01(1 - e) * clamp01(0.5 + i * 0.5) * clamp01(1 - m) * 2.2,
  },
  {
    id: 'guilty',
    coefficient: -0.107,
    hue: 0.36, sat: 0.28, light: 0.44,
    hairHue: 0.3, hairSat: 0.2, hairLight: 0.22,
    headScale: 1.0, girth: 0.94, limbLength: 1.02, heightBias: 0.94, hair: 0.35,
    leanBias: 0.3, hunchBias: 0.85, bounceBias: 0.3, speedBias: 0.55,
    gait: 'creep', behavior: 'withdraw',
    affinity: (_i, e, m) => e * clamp01(1 - m) * 2.0,
  },
  {
    id: 'afraid',
    coefficient: -0.1291,
    hue: 0.72, sat: 0.5, light: 0.56,
    hairHue: 0.7, hairSat: 0.4, hairLight: 0.3,
    headScale: 1.2, girth: 0.82, limbLength: 0.9, heightBias: 0.86, hair: 0.75,
    leanBias: 0.2, hunchBias: 0.6, bounceBias: 1.1, speedBias: 1.6,
    gait: 'run', behavior: 'flee',
    affinity: (i, e, m) => e * m * clamp01(0.4 + i * 0.6) * 3.0,
  },
];

/** Small floor so no emotion ever fully vanishes — mixtures, not winners. */
const FLOOR = 0.05;

/** Resting proportion from the coefficient, shifted so all are positive. */
const MIN_COEFF = Math.min(...ARCHETYPES.map((a) => a.coefficient));
const RESTING = ARCHETYPES.map((a) => a.coefficient - MIN_COEFF + 0.06);

/**
 * Target headcount per emotion. Proportional, so the mix slides continuously
 * as the person changes rather than switching between states.
 */
export function distribution(
  total: number,
  intensity: number,
  effort: number,
  movement: number
): number[] {
  const weights = ARCHETYPES.map(
    (a, i) => FLOOR + RESTING[i] * (0.35 + a.affinity(intensity, effort, movement))
  );
  const sum = weights.reduce((x, y) => x + y, 0) || 1;
  return weights.map((w) => (w / sum) * total);
}
