/**
 * Archetypes.
 *
 * These are NOT labels the system assigns you. The crowd is a POPULATION —
 * affect drives a soft distribution over archetypes, and what you read is the
 * shifting mix, not a winner. 40 bright ones, 12 heavy ones and 8 sharp ones
 * milling together is a mixture, and mixtures are how feeling actually works.
 *
 * That's what keeps this continuous while the figures stay discrete: the
 * proportions move smoothly even though each body is one thing or another.
 * It also renders ambivalence honestly — bittersweet is both populations
 * present at once, which no single label can show.
 *
 * `affinity` maps the continuous channels to how many of this archetype the
 * crowd wants right now. Never thresholded, never argmax'd.
 */

export interface Archetype {
  id: string;
  /** HSL, the archetype's own palette. */
  hue: number;
  sat: number;
  light: number;
  /** Silhouette. Distinct proportions read faster than distinct colour. */
  headScale: number;
  girth: number;
  limbLength: number;
  heightBias: number;
  /** A spiky crest, for silhouette variety. 0 = none. */
  crest: number;
  /** Posture offsets layered on top of the crowd-wide posture. */
  leanBias: number;
  hunchBias: number;
  bounceBias: number;
  strideBias: number;
  /** How many of these the crowd wants, given current affect. Unnormalised. */
  affinity(intensity: number, effort: number, movement: number): number;
}

/** Small floor so no archetype ever vanishes completely — mixtures, not winners. */
const FLOOR = 0.04;

export const ARCHETYPES: Archetype[] = [
  {
    // Bright, light, springy. Lots of air under the feet.
    id: 'bright',
    hue: 0.13,
    sat: 0.78,
    light: 0.62,
    headScale: 1.15,
    girth: 0.95,
    limbLength: 1,
    heightBias: 1,
    crest: 0.6,
    leanBias: -0.12,
    hunchBias: -0.25,
    bounceBias: 1.9,
    strideBias: 1.15,
    affinity: (i, e) => FLOOR + Math.max(0, i * (1 - e)) * 1.6,
  },
  {
    // Compact, forward, hard-edged. High energy with tension behind it.
    id: 'sharp',
    hue: 0.99,
    sat: 0.72,
    light: 0.5,
    headScale: 0.92,
    girth: 1.25,
    limbLength: 0.92,
    heightBias: 0.98,
    crest: 0.9,
    leanBias: 0.3,
    hunchBias: 0.35,
    bounceBias: 0.7,
    strideBias: 1.25,
    affinity: (i, e) => FLOOR + Math.max(0, i * e) * 1.8,
  },
  {
    // Heavy, slow, folded down. Long limbs, dropped head.
    id: 'heavy',
    hue: 0.6,
    sat: 0.42,
    light: 0.42,
    headScale: 1.02,
    girth: 0.85,
    limbLength: 1.12,
    heightBias: 0.96,
    crest: 0,
    leanBias: 0.22,
    hunchBias: 0.7,
    bounceBias: 0.25,
    strideBias: 0.6,
    affinity: (i, _e, m) => FLOOR + Math.max(0, (1 - i) * (1 - m)) * 1.5,
  },
  {
    // Small, quick, skittish. Never still.
    id: 'skittish',
    hue: 0.75,
    sat: 0.55,
    light: 0.58,
    headScale: 1.22,
    girth: 0.8,
    limbLength: 0.88,
    heightBias: 0.86,
    crest: 0.35,
    leanBias: 0.1,
    hunchBias: 0.4,
    bounceBias: 1.3,
    strideBias: 0.85,
    affinity: (i, e, m) => FLOOR + Math.max(0, m * (0.4 + e * 0.6) * (1 - i * 0.3)) * 1.6,
  },
  {
    // Broad, settled, unhurried. The crowd's baseline when nothing much is
    // happening — which is most of the time, and should look like something.
    id: 'settled',
    hue: 0.45,
    sat: 0.3,
    light: 0.55,
    headScale: 1,
    girth: 1.1,
    limbLength: 1,
    heightBias: 1.04,
    crest: 0,
    leanBias: 0,
    hunchBias: 0,
    bounceBias: 0.6,
    strideBias: 0.8,
    affinity: (i, e, m) => FLOOR + Math.max(0, (1 - i) * (1 - e) * (1 - m * 0.5)) * 1.3,
  },
];

/**
 * Target headcount per archetype for a given affect and crowd size.
 * Proportional, so the mix slides continuously as the person changes.
 */
export function distribution(
  total: number,
  intensity: number,
  effort: number,
  movement: number
): number[] {
  const weights = ARCHETYPES.map((a) => Math.max(0, a.affinity(intensity, effort, movement)));
  const sum = weights.reduce((x, y) => x + y, 0) || 1;
  return weights.map((w) => (w / sum) * total);
}
