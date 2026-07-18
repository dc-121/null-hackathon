/**
 * THE CONTRACT.
 *
 * Everything in the app reads and writes this shape. Agree changes here first —
 * if you edit this file, say so in the groupchat, because everyone builds
 * against it.
 *
 * Rule that keeps the app at 60fps: this store is a plain mutable object, NOT
 * React state. The audio loop writes it, the render loop reads it, React never
 * re-renders because of it. Anything you put in `useState` at frame rate will
 * jank the whole thing.
 */

/** Which side of the conversation a signal belongs to. */
export type Side = 'user' | 'model';

/**
 * Continuous, baseline-relative affect. Every value is a z-score against the
 * speaker's OWN rolling baseline, roughly clamped to [-3, 3].
 *
 * Deliberately NOT emotion labels. 0 means "your normal", not "neutral".
 */
export interface Affect {
  /** Activation. Energy + pitch height + speech rate. The arousal axis. */
  intensity: number;
  /** Vocal effort / strain, from spectral tilt. Pushing hard vs relaxed. */
  effort: number;
  /** Pitch movement. Flat monotone (low) vs animated, varied contour (high). */
  movement: number;
  /** Are we currently hearing voiced speech at all? */
  voiced: boolean;
  /** 0..1 — how settled the rolling baseline is. Below ~0.6, don't trust the rest. */
  calibration: number;
}

export const neutralAffect = (): Affect => ({
  intensity: 0,
  effort: 0,
  movement: 0,
  voiced: false,
  calibration: 0,
});

/**
 * A single prominence event — one syllable that stood out from the speaker's
 * baseline. Emitted by the audio loop, consumed by the crowd + word layers.
 *
 * Fires ~200ms after the syllable. Words attach later (see `word`), when ASR
 * catches up — never gate the visuals on that.
 */
export interface Emphasis {
  id: number;
  side: Side;
  /** performance.now() at detection. */
  at: number;
  /** How far above baseline this syllable landed. Roughly 0..3. */
  weight: number;
  /** Attached retroactively once transcription resolves. */
  word?: string;
}

/** Per-side live state. */
export interface SideState {
  affect: Affect;
  /** Recent prominence events, newest last. Trimmed to the last few seconds. */
  emphases: Emphasis[];
}

export interface EmotionStore {
  user: SideState;
  model: SideState;
  /** Rises as the two sides' affect converges. 0 = divergent, 1 = in sync. */
  attunement: number;
}

/** The single mutable instance. Import it, mutate it, read it in your loop. */
export const store: EmotionStore = {
  user: { affect: neutralAffect(), emphases: [] },
  model: { affect: neutralAffect(), emphases: [] },
  attunement: 0,
};

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
