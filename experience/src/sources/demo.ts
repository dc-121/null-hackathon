import {
  clearEmotionModality,
  emitFrame,
  normalizeEmotionScores,
  setEmotionModality,
  type EmotionScores,
} from '../state/emotion.js';

export interface DemoSignalUpdate {
  face: EmotionScores;
  prosody: EmotionScores;
  faceConfidence: number;
  prosodyConfidence: number;
}

interface DemoSignalOptions {
  face?: boolean;
  prosody?: boolean;
}

/**
 * Deterministic, opt-in-only source for stage rehearsal when hardware or the
 * local face service is unavailable. The caller is responsible for displaying
 * a permanent DEMO SIGNAL badge while this runs.
 */
export function startDemoSignals(
  onUpdate: (update: DemoSignalUpdate) => void,
  { face: useFace = true, prosody: useProsody = true }: DemoSignalOptions = {}
): () => void {
  const started = performance.now();
  const tick = () => {
    const t = (performance.now() - started) / 1000;
    const pulse = (offset: number, speed = 1) => (Math.sin(t * speed + offset) + 1) / 2;
    const face = normalizeEmotionScores({
      happy: 0.12 + pulse(0.2, 0.42) * 0.42,
      sad: 0.08 + pulse(2.8, 0.3) * 0.14,
      angry: 0.05 + pulse(4.1, 0.55) * 0.1,
      afraid: 0.06 + pulse(1.7, 0.64) * 0.11,
      surprised: 0.08 + pulse(5.2, 0.78) * 0.18,
    });
    const prosody = normalizeEmotionScores({
      happy: 0.15 + pulse(0.7, 0.5) * 0.28,
      sad: 0.1 + pulse(3.1, 0.34) * 0.15,
      angry: 0.09 + pulse(4.3, 0.6) * 0.17,
      afraid: 0.09 + pulse(2.1, 0.72) * 0.15,
      surprised: 0.11 + pulse(5.6, 0.83) * 0.2,
    });
    const faceConfidence = 0.72;
    const prosodyConfidence = 0.38;
    if (useFace) setEmotionModality('user', 'demo-face', face, faceConfidence);
    if (useProsody) {
      setEmotionModality('user', 'demo-prosody', prosody, prosodyConfidence);
      emitFrame({
        source: 'demo-prosody',
        side: 'user',
        at: performance.now(),
        confidence: 0.4,
        channels: {
          intensity: pulse(0, 0.8) * 2 - 0.4,
          effort: pulse(1.3, 0.62) * 1.6 - 0.35,
          movement: pulse(2.4, 1.05) * 1.8,
        },
      });
    }
    onUpdate({ face, prosody, faceConfidence, prosodyConfidence });
  };

  tick();
  const timer = window.setInterval(tick, 160);
  return () => {
    window.clearInterval(timer);
    if (useFace) clearEmotionModality('user', 'demo-face');
    if (useProsody) clearEmotionModality('user', 'demo-prosody');
  };
}
