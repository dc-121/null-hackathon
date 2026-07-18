import {
  SHARED_EMOTIONS,
  clearEmotionModality,
  emitFrame,
  normalizeEmotionScores,
  setEmotionModality,
  type EmotionScores,
} from '../state/emotion.js';

export interface ProsodySnapshot {
  scores: EmotionScores;
  confidence: number;
  voiced: boolean;
  rms: number;
  pitchHz: number | null;
  spectralCentroidHz: number;
  variation: number;
  calibration: number;
}

export interface ProsodyHandle {
  beginCapture(): void;
  endCapture(): { scores: EmotionScores; confidence: number };
  current(): ProsodySnapshot;
  stop(): void;
}

interface RunningBaseline {
  mean: number;
  variance: number;
  samples: number;
}

const clamp = (value: number, low: number, high: number) =>
  Math.max(low, Math.min(high, value));

const emptyScores = (): EmotionScores =>
  normalizeEmotionScores(Object.fromEntries(SHARED_EMOTIONS.map((emotion) => [emotion, 0.2])));

function updateBaseline(baseline: RunningBaseline, value: number, alpha = 0.035): number {
  if (baseline.samples === 0) {
    baseline.mean = value;
    baseline.variance = 0.04;
  } else {
    const delta = value - baseline.mean;
    baseline.mean += delta * alpha;
    baseline.variance = Math.max(0.0001, baseline.variance * (1 - alpha) + delta * delta * alpha);
  }
  baseline.samples += 1;
  return clamp((value - baseline.mean) / Math.sqrt(baseline.variance), -3, 3);
}

function estimatePitch(samples: Float32Array, sampleRate: number, rms: number): number | null {
  if (rms < 0.012) return null;
  const minLag = Math.floor(sampleRate / 420);
  const maxLag = Math.min(samples.length - 2, Math.floor(sampleRate / 75));
  let bestLag = 0;
  let bestCorrelation = 0;

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let product = 0;
    let leftPower = 0;
    let rightPower = 0;
    const count = samples.length - lag;
    for (let index = 0; index < count; index += 1) {
      const left = samples[index];
      const right = samples[index + lag];
      product += left * right;
      leftPower += left * left;
      rightPower += right * right;
    }
    const correlation = product / Math.sqrt(leftPower * rightPower + 1e-9);
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }
  return bestCorrelation >= 0.58 && bestLag ? sampleRate / bestLag : null;
}

function spectralCentroid(decibels: Float32Array, sampleRate: number): number {
  const binWidth = sampleRate / (decibels.length * 2);
  let weighted = 0;
  let total = 0;
  for (let index = 1; index < decibels.length; index += 1) {
    const frequency = index * binWidth;
    if (frequency > 8000) break;
    const db = decibels[index];
    if (!Number.isFinite(db)) continue;
    const amplitude = 10 ** (db / 20);
    weighted += frequency * amplitude;
    total += amplitude;
  }
  return total > 0 ? weighted / total : 0;
}

function conservativeEmotionMix(
  energyZ: number,
  pitchZ: number,
  centroidZ: number,
  variation: number,
  onset: number
): EmotionScores {
  // Prosody is suggestive, not definitive. Keep most of the mass near a uniform
  // prior and let confidence, separately, determine how much fusion trusts it.
  const logits: EmotionScores = {
    happy: 0.2 * Math.max(0, pitchZ) + 0.18 * variation - 0.08 * Math.max(0, centroidZ),
    sad: 0.2 * Math.max(0, -energyZ) + 0.14 * Math.max(0, -pitchZ),
    angry: 0.24 * Math.max(0, energyZ) + 0.2 * Math.max(0, centroidZ),
    afraid: 0.15 * Math.max(0, pitchZ) + 0.22 * variation + 0.08 * Math.max(0, energyZ),
    surprised: 0.28 * onset + 0.15 * Math.max(0, pitchZ) + 0.12 * variation,
  };
  const maxLogit = Math.max(...Object.values(logits));
  const exponentials = SHARED_EMOTIONS.map((emotion) => Math.exp(logits[emotion] - maxLogit));
  const total = exponentials.reduce((sum, value) => sum + value, 0) || 1;
  const mixed: Partial<EmotionScores> = {};
  SHARED_EMOTIONS.forEach((emotion, index) => {
    mixed[emotion] = 0.68 * 0.2 + 0.32 * (exponentials[index] / total);
  });
  return normalizeEmotionScores(mixed);
}

export async function startProsodySource(
  stream: MediaStream,
  onUpdate: (snapshot: ProsodySnapshot) => void
): Promise<ProsodyHandle> {
  if (!stream.getAudioTracks().length) throw new Error('No microphone track is available.');

  const context = new AudioContext({ latencyHint: 'interactive' });
  await context.resume();
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.42;
  source.connect(analyser);

  const timeData = new Float32Array(analyser.fftSize);
  const frequencyData = new Float32Array(analyser.frequencyBinCount);
  const energyBaseline: RunningBaseline = { mean: 0, variance: 0.04, samples: 0 };
  const pitchBaseline: RunningBaseline = { mean: 0, variance: 400, samples: 0 };
  const centroidBaseline: RunningBaseline = { mean: 0, variance: 90_000, samples: 0 };
  let previousLogEnergy = -6;
  let previousPitch: number | null = null;
  let timer = 0;
  let stopped = false;
  let capturing = false;
  let captureWeight = 0;
  let captureConfidence = 0;
  let captureSamples = 0;
  let captureFrames = 0;
  const captureScores: Partial<EmotionScores> = {};
  let latest: ProsodySnapshot = {
    scores: emptyScores(),
    confidence: 0,
    voiced: false,
    rms: 0,
    pitchHz: null,
    spectralCentroidHz: 0,
    variation: 0,
    calibration: 0,
  };

  const analyze = () => {
    if (stopped) return;
    analyser.getFloatTimeDomainData(timeData);
    analyser.getFloatFrequencyData(frequencyData);

    let power = 0;
    for (const sample of timeData) power += sample * sample;
    const rms = Math.sqrt(power / timeData.length);
    const logEnergy = Math.log(Math.max(rms, 1e-5));
    const pitch = estimatePitch(timeData, context.sampleRate, rms);
    const centroid = spectralCentroid(frequencyData, context.sampleRate);
    const voiced = rms > 0.012 && pitch !== null;

    let energyZ = 0;
    let pitchZ = 0;
    let centroidZ = 0;
    if (rms > 0.006) {
      energyZ = updateBaseline(energyBaseline, logEnergy);
      centroidZ = updateBaseline(centroidBaseline, centroid);
      if (pitch !== null) pitchZ = updateBaseline(pitchBaseline, pitch);
    }

    const pitchShift = pitch !== null && previousPitch !== null
      ? Math.abs(Math.log2(pitch / previousPitch)) * 2.2
      : 0;
    const energyShift = Math.abs(logEnergy - previousLogEnergy) * 0.75;
    const variation = clamp(pitchShift + energyShift, 0, 2.5);
    const onset = clamp((logEnergy - previousLogEnergy) * 1.4, 0, 2.5);
    const calibration = clamp(energyBaseline.samples / 70, 0, 1);
    const scores = conservativeEmotionMix(energyZ, pitchZ, centroidZ, variation, onset);
    const signalQuality = clamp((rms - 0.008) / 0.05, 0, 1);
    const confidence = voiced
      ? clamp(0.1 + calibration * 0.25 + signalQuality * 0.18, 0, 0.53)
      : 0.015;

    latest = {
      scores,
      confidence,
      voiced,
      rms,
      pitchHz: pitch,
      spectralCentroidHz: centroid,
      variation,
      calibration,
    };

    if (voiced) {
      setEmotionModality('user', 'voice-prosody', scores, confidence);
      emitFrame({
        source: 'voice-prosody',
        side: 'user',
        at: performance.now(),
        confidence,
        channels: {
          intensity: clamp(energyZ, -2, 3),
          effort: clamp(energyZ * 0.65 + centroidZ * 0.35, -2, 3),
          movement: clamp(variation, 0, 3),
        },
      });
    } else {
      clearEmotionModality('user', 'voice-prosody');
    }

    if (capturing) {
      captureFrames += 1;
      if (voiced) {
        const weight = Math.max(0.05, confidence);
        for (const emotion of SHARED_EMOTIONS) {
          captureScores[emotion] = (captureScores[emotion] ?? 0) + scores[emotion] * weight;
        }
        captureWeight += weight;
        captureConfidence += confidence;
        captureSamples += 1;
      }
    }
    onUpdate(latest);

    previousLogEnergy = logEnergy;
    if (pitch !== null) previousPitch = pitch;
    timer = window.setTimeout(analyze, 90);
  };

  analyze();
  return {
    beginCapture() {
      capturing = true;
      captureWeight = 0;
      captureConfidence = 0;
      captureSamples = 0;
      captureFrames = 0;
      for (const emotion of SHARED_EMOTIONS) captureScores[emotion] = 0;
    },
    endCapture() {
      capturing = false;
      if (!captureSamples || !captureWeight) {
        return { scores: latest.scores, confidence: latest.voiced ? latest.confidence : 0 };
      }
      const average: Partial<EmotionScores> = {};
      for (const emotion of SHARED_EMOTIONS) {
        average[emotion] = (captureScores[emotion] ?? 0) / captureWeight;
      }
      return {
        scores: normalizeEmotionScores(average),
        // A momentary voiced/noisy frame in a mostly silent recording should
        // not carry the same weight as sustained speech.
        confidence: clamp(
          (captureConfidence / captureSamples) *
            Math.sqrt(captureSamples / Math.max(1, captureFrames)),
          0,
          0.53
        ),
      };
    },
    current: () => latest,
    stop() {
      stopped = true;
      window.clearTimeout(timer);
      clearEmotionModality('user', 'voice-prosody');
      source.disconnect();
      analyser.disconnect();
      void context.close();
    },
  };
}
