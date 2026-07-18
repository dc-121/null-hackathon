import {
  clearEmotionModality,
  emitFrame,
  normalizeEmotionScores,
  setEmotionModality,
  type Emotion,
  type EmotionScores,
} from '../state/emotion.js';

export const DEMO_PROOF_LINE = "I'm fine. Really, I'm fine.";
export const DEFAULT_DEMO_EMOTION: Emotion = 'sad';

export interface DemoSignalUpdate {
  face: EmotionScores;
  prosody: EmotionScores;
  faceConfidence: number;
  prosodyConfidence: number;
}

interface DemoSignalOptions {
  face?: boolean;
  prosody?: boolean;
  emotion?: () => Emotion;
}

const MOTION_PROFILES: Record<
  Emotion,
  { intensity: number; effort: number; movement: number; valence: number }
> = {
  happy: { intensity: 1.25, effort: 0.45, movement: 1.35, valence: 1.4 },
  sad: { intensity: -0.35, effort: -0.2, movement: 0.22, valence: -1.25 },
  angry: { intensity: 1.65, effort: 1.55, movement: 1.05, valence: -1.3 },
  afraid: { intensity: 1.25, effort: 1.05, movement: 1.55, valence: -1.0 },
  surprised: { intensity: 1.45, effort: 0.65, movement: 1.7, valence: 0.25 },
};

function peakedScores(emotion: Emotion, peak: number): EmotionScores {
  const remainder = (1 - peak) / 4;
  return normalizeEmotionScores({
    happy: emotion === 'happy' ? peak : remainder,
    sad: emotion === 'sad' ? peak : remainder,
    angry: emotion === 'angry' ? peak : remainder,
    afraid: emotion === 'afraid' ? peak : remainder,
    surprised: emotion === 'surprised' ? peak : remainder,
  });
}

/** One named, stable rehearsal signal used by both the UI and request payload. */
export function demoSignalSnapshot(emotion: Emotion): DemoSignalUpdate {
  return {
    face: peakedScores(emotion, 0.86),
    prosody: peakedScores(emotion, 0.78),
    faceConfidence: 0.86,
    prosodyConfidence: 0.68,
  };
}

/**
 * Deterministic, opt-in-only source for stage rehearsal when hardware or the
 * local face service is unavailable. The caller is responsible for displaying
 * a permanent DEMO SIGNAL badge while this runs.
 */
export function startDemoSignals(
  onUpdate: (update: DemoSignalUpdate) => void,
  {
    face: useFace = true,
    prosody: useProsody = true,
    emotion: readEmotion = () => 'happy',
  }: DemoSignalOptions = {}
): () => void {
  const started = performance.now();
  const tick = () => {
    const t = (performance.now() - started) / 1000;
    const selectedEmotion = readEmotion();
    const { face, prosody, faceConfidence, prosodyConfidence } = (
      demoSignalSnapshot(selectedEmotion)
    );
    if (useFace) setEmotionModality('user', 'demo-face', face, faceConfidence);
    if (useProsody) {
      setEmotionModality('user', 'demo-prosody', prosody, prosodyConfidence);
      const motion = MOTION_PROFILES[selectedEmotion];
      const breathe = Math.sin(t * 2.1) * 0.12;
      emitFrame({
        source: `demo-prosody-${selectedEmotion}`,
        side: 'user',
        at: performance.now(),
        confidence: 0.68,
        channels: {
          intensity: motion.intensity + breathe,
          effort: motion.effort + breathe * 0.5,
          movement: Math.max(0, motion.movement + breathe),
          valence: motion.valence,
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
