import {
  SHARED_EMOTIONS,
  normalizeEmotionScores,
  type Emotion,
  type EmotionScores,
} from '../state/emotion.js';

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
 *
 * When a shared-five classifier is active, its explicit probability mix takes
 * over the population target. The richer nine-archetype baseline remains for
 * the idle simulation, but unsupported archetypes never impersonate a live
 * detected label.
 */

export type Gait =
  | 'walk' | 'skip' | 'run' | 'trudge' | 'creep' | 'stagger' | 'drift'
  | 'stomp' | 'scurry';
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

  // palette — `hue/sat/light` is the SHIRT, which is the biggest area and so
  // carries the emotion at distance. Skin stays a neutral tone so the crowd
  // reads as people rather than as coloured markers.
  hue: number;
  sat: number;
  light: number;
  hairHue: number;
  hairSat: number;
  hairLight: number;
  /** Skin tone lightness. Hue/sat are shared so the crowd looks like one species. */
  skin: number;

  // face — these do more work than anything else at this size
  /** -1 frown, 0 flat, +1 smile. */
  mouth: number;
  /** 0 closed, 1 wide open. */
  mouthOpen: number;
  /** -1 furrowed down (angry), +1 raised inner (worried). */
  brow: number;

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
    skin: 0.7, mouth: 0.25, mouthOpen: 0, brow: 0,
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
    skin: 0.62, mouth: -0.5, mouthOpen: 0.55, brow: 0.7,
    headScale: 1.02, girth: 0.92, limbLength: 1.04, heightBias: 0.99, hair: 0.7,
    leanBias: 0.34, hunchBias: 0.3, bounceBias: 1.3, speedBias: 1.9,
    gait: 'run', behavior: 'frantic',
    affinity: (i, e, m) => i * e * clamp01(0.4 + m) * 3.2,
  },
  {
    id: 'surprised',
    coefficient: 0.0338,
    // Cyan is intentionally distinct from happy's gold so every shared
    // emotion remains identifiable by colour at crowd scale.
    hue: 0.52, sat: 0.76, light: 0.61,
    hairHue: 0.54, hairSat: 0.58, hairLight: 0.38,
    skin: 0.74, mouth: 0, mouthOpen: 1, brow: 1,
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
    skin: 0.58, mouth: -0.9, mouthOpen: 0.25, brow: -1,
    headScale: 0.94, girth: 1.3, limbLength: 0.94, heightBias: 1.03, hair: 0.9,
    leanBias: 0.42, hunchBias: 0.4, bounceBias: 0.8, speedBias: 1.45,
    gait: 'stomp', behavior: 'strike',
    affinity: (i, e) => i * e * e * 3.6,
  },
  {
    id: 'happy',
    coefficient: -0.0419,
    hue: 0.14, sat: 0.85, light: 0.62,
    hairHue: 0.1, hairSat: 0.7, hairLight: 0.5,
    skin: 0.72, mouth: 1, mouthOpen: 0.3, brow: 0.15,
    headScale: 1.14, girth: 0.98, limbLength: 1.0, heightBias: 1.0, hair: 0.6,
    leanBias: -0.16, hunchBias: -0.35, bounceBias: 2.4, speedBias: 1.2,
    gait: 'skip', behavior: 'approach',
    affinity: (i, e) => i * clamp01(1 - e) * 3.4,
  },
  {
    id: 'sad',
    coefficient: -0.0631,
    // Clear cobalt-blue clothing stays recognizable against the near-black
    // floor. Posture and gait communicate sadness; darkness should not make
    // the people themselves disappear.
    hue: 0.59, sat: 0.78, light: 0.58,
    hairHue: 0.61, hairSat: 0.5, hairLight: 0.34,
    skin: 0.7, mouth: -0.85, mouthOpen: 0, brow: 0.85,
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
    skin: 0.74, mouth: 0.8, mouthOpen: 0, brow: 0.3,
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
    skin: 0.6, mouth: -0.45, mouthOpen: 0, brow: 0.5,
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
    skin: 0.68, mouth: -0.6, mouthOpen: 0.7, brow: 1,
    headScale: 1.2, girth: 0.82, limbLength: 0.9, heightBias: 0.86, hair: 0.75,
    leanBias: 0.2, hunchBias: 0.6, bounceBias: 1.1, speedBias: 1.6,
    gait: 'scurry', behavior: 'flee',
    affinity: (i, e, m) => e * m * clamp01(0.4 + i * 0.6) * 3.0,
  },
];

/** Small floor so no emotion ever fully vanishes — mixtures, not winners. */
const FLOOR = 0.05;

/** Resting proportion from the coefficient, shifted so all are positive. */
const MIN_COEFF = Math.min(...ARCHETYPES.map((a) => a.coefficient));
const RESTING = ARCHETYPES.map((a) => a.coefficient - MIN_COEFF + 0.06);

const SHARED_EMOTION_IDS = new Set<string>(SHARED_EMOTIONS);

export interface ExplicitDistribution {
  scores: EmotionScores;
  confidence: number;
  active: boolean;
}

/**
 * A categorical source may expose only part of its probability mass after its
 * richer taxonomy is projected onto the shared five. Keep that missing mass
 * visible as uncertainty (an even spread over the five), and use confidence
 * to temper unsupported certainty. Crucially, no active target population is
 * assigned to visual-only archetypes such as calm, desperate, loving, or
 * guilty.
 */
function explicitDistribution(total: number, explicit: ExplicitDistribution): number[] {
  const scores = normalizeEmotionScores(explicit.scores);
  const confidence = clamp01(explicit.confidence);
  const representedMass = SHARED_EMOTIONS.reduce(
    (sum, emotion) => sum + scores[emotion] * confidence,
    0
  );
  const unresolvedPerEmotion = Math.max(0, 1 - representedMass) / SHARED_EMOTIONS.length;

  return ARCHETYPES.map((archetype) => {
    if (!SHARED_EMOTION_IDS.has(archetype.id)) return 0;
    const emotion = archetype.id as Emotion;
    return (scores[emotion] * confidence + unresolvedPerEmotion) * total;
  });
}

/**
 * Target headcount per emotion. Proportional, so the mix slides continuously
 * as the person changes rather than switching between states.
 */
export function distribution(
  total: number,
  intensity: number,
  effort: number,
  movement: number,
  explicit?: ExplicitDistribution
): number[] {
  if (explicit?.active) return explicitDistribution(total, explicit);

  const weights = ARCHETYPES.map(
    (a, i) => FLOOR + RESTING[i] * (0.35 + a.affinity(intensity, effort, movement))
  );
  const sum = weights.reduce((x, y) => x + y, 0) || 1;
  return weights.map((w) => (w / sum) * total);
}
