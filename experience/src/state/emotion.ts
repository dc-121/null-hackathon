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

export interface SideState {
  affect: Affect;
  /** Recent prominence events, newest last. Trimmed each frame. */
  emphases: Emphasis[];
}

export interface EmotionStore {
  user: SideState;
  model: SideState;
  /** Rises as the two sides converge. 0 = divergent, 1 = in sync. */
  attunement: number;
}

/** The single mutable instance. Import it, read it in your loop. */
export const store: EmotionStore = {
  user: { affect: neutralAffect(), emphases: [] },
  model: { affect: neutralAffect(), emphases: [] },
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
  const a = store.user.affect;
  const b = store.model.affect;
  const d =
    Math.abs(a.intensity - b.intensity) +
    Math.abs(a.effort - b.effort) +
    Math.abs(a.movement - b.movement);
  const target = Math.max(0, 1 - d / 6);
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
