/**
 * THE CONTRACT.
 *
 * Everything in the app reads and writes these shapes. Agree changes here
 * first — if you edit this file, say so in the groupchat, because all four
 * workstreams build against it.
 *
 * Rule that keeps the app at 60fps: this store is a plain mutable object, NOT
 * React state. Sources write it, render loops read it, React never re-renders
 * because of it. Anything you put in `useState` at frame rate will jank
 * everything.
 */

/** Which side of the conversation a signal belongs to. */
export type Side = 'user' | 'model';

/**
 * The categorical contract shared by face, voice, language model, and crowd.
 * Keep this deliberately small: adapters must map their richer taxonomies
 * into these labels rather than leaking source-specific labels downstream.
 */
export const SHARED_EMOTIONS = ['happy', 'sad', 'angry', 'afraid', 'surprised'] as const;

export type Emotion = (typeof SHARED_EMOTIONS)[number];

/**
 * Scores are probability MASS, not necessarily a complete distribution.
 * A sum below 1 means the source assigned some probability to labels outside
 * the shared five. Consumers must preserve that uncertainty.
 */
export type EmotionScores = Record<Emotion, number>;

const emptyEmotionScores = (): EmotionScores => ({
  happy: 0,
  sad: 0,
  angry: 0,
  afraid: 0,
  surprised: 0,
});

/**
 * Sanitize shared-five scores without turning omitted/excluded mass into
 * certainty. Values are clamped at zero; only an over-full (>1) distribution
 * is scaled down. A valid partial distribution is never scaled up.
 */
export function normalizeEmotionScores(
  scores: Partial<Record<Emotion, number>>
): EmotionScores {
  const normalized = emptyEmotionScores();
  let total = 0;

  for (const emotion of SHARED_EMOTIONS) {
    const value = scores[emotion];
    const safe = typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : 0;
    normalized[emotion] = safe;
    total += safe;
  }

  if (total > 1) {
    for (const emotion of SHARED_EMOTIONS) normalized[emotion] /= total;
  }

  return normalized;
}

/** Highest supported score, or null when there is no supported mass. */
export function dominantEmotion(
  scores: Partial<Record<Emotion, number>>
): Emotion | null {
  const normalized = normalizeEmotionScores(scores);
  let winner: Emotion | null = null;
  let best = 0;

  for (const emotion of SHARED_EMOTIONS) {
    if (normalized[emotion] > best) {
      best = normalized[emotion];
      winner = emotion;
    }
  }

  return winner;
}

/**
 * The measurable axes. Deliberately NOT emotion labels — every value is a
 * z-score against that speaker's OWN rolling baseline, roughly [-3, 3].
 * 0 means "your normal", not "neutral".
 */
export type Channel = 'intensity' | 'effort' | 'movement' | 'valence';

export const CHANNELS: Channel[] = ['intensity', 'effort', 'movement', 'valence'];

/**
 * One reading from one source.
 *
 * `channels` is SPARSE and that's the point — fill only what you can actually
 * measure. Heart rate fills `intensity` and nothing else, because HR genuinely
 * cannot tell excited from anxious. Let the type say so.
 */
export interface SignalFrame {
  source: string;
  side: Side;
  /** performance.now() at measurement. */
  at: number;
  /** 0..1. How much this reading should count in the merge. */
  confidence: number;
  channels: Partial<Record<Channel, number>>;
}

/**
 * Every input wraps to this, local or remote. The consumer can't tell which is
 * which — that's what lets us swap Hume for openSMILE for a local model
 * without touching the visuals.
 */
export interface SourceAdapter {
  id: string;
  /** Which channels this source can actually speak to. */
  provides: Channel[];
  /** Begin emitting. Resolves to a stop function. */
  start(emit: (frame: SignalFrame) => void): Promise<() => void>;
}

/** Fused, current affect for one side. What the visuals read. */
export interface Affect {
  intensity: number;
  effort: number;
  movement: number;
  valence: number;
  /** Is this side currently producing signal at all? */
  active: boolean;
  /** 0..1 — how settled the baselines are. Below ~0.6, don't trust the rest. */
  calibration: number;
}

export const neutralAffect = (): Affect => ({
  intensity: 0,
  effort: 0,
  movement: 0,
  valence: 0,
  active: false,
  calibration: 0,
});

/**
 * A single prominence event — one syllable that stood out from baseline.
 * Fires ~200ms after the syllable, with no ASR involved. Words attach later
 * via `word` when transcription catches up; never gate visuals on that.
 */
export interface Emphasis {
  id: number;
  side: Side;
  at: number;
  /** How far above baseline this syllable landed. Roughly 0..3. */
  weight: number;
  word?: string;
}

/** One timestamped categorical reading from one face/voice/model source. */
export interface EmotionReading {
  source: string;
  scores: EmotionScores;
  /** 0..1 reliability of this source reading. */
  confidence: number;
  /** performance.now() at measurement. */
  at: number;
  /** How long the reading remains eligible for fusion. */
  ttlMs: number;
}

/** Smoothed categorical result consumed by the crowd and UI. */
export interface ExplicitEmotion {
  scores: EmotionScores;
  confidence: number;
  active: boolean;
}

const neutralExplicitEmotion = (): ExplicitEmotion => ({
  scores: emptyEmotionScores(),
  confidence: 0,
  active: false,
});

const emptyEmotionModalities = (): Record<string, EmotionReading> =>
  Object.create(null) as Record<string, EmotionReading>;

export interface SideState {
  affect: Affect;
  /** Recent prominence events, newest last. Trimmed each frame. */
  emphases: Emphasis[];
  /** Latest live reading per modality/source (face, voice, etc.). */
  emotionModalities: Record<string, EmotionReading>;
  /** Phrase-level model output, kept separately because it has a longer TTL. */
  directEmotion: EmotionReading | null;
  /** Confidence-weighted, smoothed fusion of all current categorical readings. */
  emotion: ExplicitEmotion;
}

export interface EmotionStore {
  user: SideState;
  model: SideState;
  /** Rises as the two sides converge. 0 = divergent, 1 = in sync. */
  attunement: number;
}

/** The single mutable instance. Import it, read it in your loop. */
export const store: EmotionStore = {
  user: {
    affect: neutralAffect(),
    emphases: [],
    emotionModalities: emptyEmotionModalities(),
    directEmotion: null,
    emotion: neutralExplicitEmotion(),
  },
  model: {
    affect: neutralAffect(),
    emphases: [],
    emotionModalities: emptyEmotionModalities(),
    directEmotion: null,
    emotion: neutralExplicitEmotion(),
  },
  attunement: 0,
};

// --- ingestion -----------------------------------------------------------
// Frames are ADDITIVE, never authoritative. No source writes `affect`
// directly — everyone emits, fusion decides. The moment one source mutates
// affect itself you have a race and two people editing the same lines.

const inbox: SignalFrame[] = [];

/** Every source calls this. That's the whole API. */
export function emitFrame(frame: SignalFrame): void {
  inbox.push(frame);
}

/**
 * How long a frame stays relevant. Rates differ by orders of magnitude
 * (prosody 60Hz, face 30Hz, HR 1Hz, utterance-level ~1s), so weight has to
 * decay with age — otherwise a four-second-old reading fights live prosody
 * and the visuals feel wrong in ways that are miserable to debug.
 */
const FRAME_TTL_MS = 2000;

/** Live face/voice classifiers should disappear promptly when input stops. */
const EMOTION_MODALITY_TTL_MS = 2000;

/** Long enough to cover normal model response playback; setting it refreshes it. */
const DIRECT_EMOTION_TTL_MS = 30_000;

const clamp01 = (value: number): number =>
  Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;

function makeEmotionReading(
  source: string,
  scores: Partial<Record<Emotion, number>>,
  confidence: number,
  ttlMs: number
): EmotionReading {
  return {
    source,
    scores: normalizeEmotionScores(scores),
    confidence: clamp01(confidence),
    at: performance.now(),
    ttlMs,
  };
}

/**
 * Set or refresh one live categorical modality. Reusing `source` replaces its
 * previous frame, so 30/60fps classifiers do not grow an unbounded inbox.
 */
export function setEmotionModality(
  side: Side,
  source: string,
  scores: Partial<Record<Emotion, number>>,
  confidence: number
): void {
  store[side].emotionModalities[source] = makeEmotionReading(
    source,
    scores,
    confidence,
    EMOTION_MODALITY_TTL_MS
  );
}

export function clearEmotionModality(side: Side, source: string): void {
  delete store[side].emotionModalities[source];
}

/** Set or refresh phrase-level categorical emotion (normally model output). */
export function setDirectEmotion(
  side: Side,
  scores: Partial<Record<Emotion, number>>,
  confidence: number
): void {
  store[side].directEmotion = makeEmotionReading(
    'direct',
    scores,
    confidence,
    DIRECT_EMOTION_TTL_MS
  );
}

export function clearDirectEmotion(side: Side): void {
  store[side].directEmotion = null;
}

function readingFreshness(reading: EmotionReading, now: number): number {
  return Math.max(0, 1 - Math.max(0, now - reading.at) / reading.ttlMs);
}

/** Fuse categorical sources without reallocating their excluded label mass. */
function fuseExplicitEmotion(side: Side, now: number, smoothing: number): void {
  const sideState = store[side];
  const readings: EmotionReading[] = [];

  if (sideState.directEmotion) {
    if (now - sideState.directEmotion.at >= sideState.directEmotion.ttlMs) {
      sideState.directEmotion = null;
    } else {
      // A direct reading is already the authoritative fused result for a turn
      // (or one measured response phrase). Mixing the live inputs back into it
      // would count the same face/prosody evidence twice.
      readings.push(sideState.directEmotion);
    }
  }

  if (!sideState.directEmotion) {
    for (const [source, reading] of Object.entries(sideState.emotionModalities)) {
      if (now - reading.at >= reading.ttlMs) {
        delete sideState.emotionModalities[source];
        continue;
      }
      readings.push(reading);
    }
  }

  const sums = emptyEmotionScores();
  let scoreWeight = 0;
  let confidenceMiss = 1;
  let hasSupportedMass = false;

  for (const reading of readings) {
    const freshness = readingFreshness(reading, now);
    const weight = reading.confidence * freshness;
    if (weight <= 0) continue;

    scoreWeight += weight;
    confidenceMiss *= 1 - weight;
    for (const emotion of SHARED_EMOTIONS) {
      const score = reading.scores[emotion];
      sums[emotion] += score * weight;
      if (score > 0) hasSupportedMass = true;
    }
  }

  const target = emptyEmotionScores();
  if (scoreWeight > 0) {
    for (const emotion of SHARED_EMOTIONS) target[emotion] = sums[emotion] / scoreWeight;
  }

  const emotionState = sideState.emotion;
  const amount = clamp01(smoothing);
  for (const emotion of SHARED_EMOTIONS) {
    emotionState.scores[emotion] += (target[emotion] - emotionState.scores[emotion]) * amount;
  }
  emotionState.confidence +=
    (clamp01(1 - confidenceMiss) - emotionState.confidence) * amount;
  emotionState.active = scoreWeight > 0.001 && hasSupportedMass;
}

/** Confidence-weighted merge of whatever arrived, with staleness decay. */
export function fuse(smoothing = 0.15): void {
  const now = performance.now();

  while (inbox.length && now - inbox[0].at > FRAME_TTL_MS) inbox.shift();

  for (const side of ['user', 'model'] as const) {
    const sums: Record<string, number> = {};
    const weights: Record<string, number> = {};
    let anyWeight = 0;

    for (const frame of inbox) {
      if (frame.side !== side) continue;
      const age = now - frame.at;
      if (age > FRAME_TTL_MS) continue;
      const w = frame.confidence * (1 - age / FRAME_TTL_MS);
      if (w <= 0) continue;

      for (const [channel, value] of Object.entries(frame.channels)) {
        if (value === undefined) continue;
        sums[channel] = (sums[channel] ?? 0) + value * w;
        weights[channel] = (weights[channel] ?? 0) + w;
      }
      anyWeight += w;
    }

    const affect = store[side].affect;
    for (const channel of CHANNELS) {
      const target = weights[channel] ? sums[channel] / weights[channel] : 0;
      affect[channel] += (target - affect[channel]) * smoothing;
    }
    affect.active = anyWeight > 0.05;

    fuseExplicitEmotion(side, now, smoothing);
  }
}

let nextEmphasisId = 1;

export function pushEmphasis(side: Side, weight: number, word?: string): Emphasis {
  const e: Emphasis = { id: nextEmphasisId++, side, at: performance.now(), weight, word };
  store[side].emphases.push(e);
  return e;
}

/** Drop emphasis events older than `maxAgeMs`. Call once per frame. */
export function trimEmphases(maxAgeMs = 4000): void {
  const cutoff = performance.now() - maxAgeMs;
  for (const side of ['user', 'model'] as const) {
    const list = store[side].emphases;
    let i = 0;
    while (i < list.length && list[i].at < cutoff) i++;
    if (i > 0) list.splice(0, i);
  }
}

/**
 * Attunement = how closely the model's affect tracks the user's, smoothed.
 * This is what makes empathy visible: when it lands, the two crowds sync.
 */
export function updateAttunement(smoothing = 0.02): void {
  const userEmotion = store.user.emotion;
  const modelEmotion = store.model.emotion;
  let target = 0;

  if (userEmotion.active && modelEmotion.active) {
    const distance = SHARED_EMOTIONS.reduce(
      (sum, emotion) =>
        sum + Math.abs(userEmotion.scores[emotion] - modelEmotion.scores[emotion]),
      0
    ) / 2;
    const jointConfidence = Math.sqrt(
      clamp01(userEmotion.confidence) * clamp01(modelEmotion.confidence)
    );
    target = clamp01(1 - distance) * jointConfidence;
  } else {
    const a = store.user.affect;
    const b = store.model.affect;
    if (a.active && b.active) {
      const distance =
        Math.abs(a.intensity - b.intensity) +
        Math.abs(a.effort - b.effort) +
        Math.abs(a.movement - b.movement);
      target = Math.max(0, 1 - distance / 6);
    }
  }
  store.attunement += (target - store.attunement) * smoothing;
}

/** Register a set of sources. Returns a stop-everything function. */
export async function startSources(adapters: SourceAdapter[]): Promise<() => void> {
  const stops = await Promise.all(
    adapters.map(async (a) => {
      try {
        return await a.start(emitFrame);
      } catch (err) {
        console.warn(`[sources] ${a.id} failed to start`, err);
        return () => {};
      }
    })
  );
  return () => stops.forEach((stop) => stop());
}
