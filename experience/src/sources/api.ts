import {
  SHARED_EMOTIONS,
  dominantEmotion,
  normalizeEmotionScores,
  type Emotion,
  type EmotionScores,
} from '../state/emotion.js';

export interface FaceAnalysis {
  detected: boolean;
  scores: EmotionScores;
  retainedMass: number;
  confidence: number;
  dominant: Emotion | null;
  detail?: string;
}

export interface ConversationPhrase {
  text: string;
  emotion: Emotion | null;
  scores: EmotionScores;
  intensity: number;
  evidence: number;
  sharedMass: number;
  direction: string;
  startSeconds: number | null;
  endSeconds: number | null;
}

export interface ResponseContext {
  dominant: Emotion | null;
  confidence: number;
  nonverbalWeight: number;
  nonverbalShift: number | null;
  nonverbalDominant: Emotion | null;
  languageDominant: Emotion | null;
  effect: 'words-only' | 'reinforced' | 'adjusted' | 'shifted' | 'nonverbal-only' | 'language-mixed' | 'mixed';
  sources: Array<'face' | 'prosody'>;
  sourceDominants: Partial<Record<'face' | 'prosody', Emotion | null>>;
  strategy: 'celebrate' | 'support' | 'de-escalate' | 'reassure' | 'orient' | 'stay-curious';
}

export interface ConversationModality {
  scores?: EmotionScores;
  confidence?: number;
  available?: boolean;
}

export interface ConversationResponse {
  transcript: string;
  human: {
    scores: EmotionScores;
    dominant: Emotion | null;
    confidence: number;
    modalities: Record<string, ConversationModality>;
  };
  responseContext: ResponseContext | null;
  response: string;
  speechId: string | null;
  phrases: ConversationPhrase[];
  timings: Record<string, number>;
}

export interface ConversationRequest {
  transcript?: string;
  audio_base64?: string;
  audio_content_type?: string;
  face_scores?: EmotionScores;
  face_confidence?: number;
  prosody_scores?: EmotionScores;
  prosody_confidence?: number;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const finite = (value: unknown, fallback = 0): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const clamp01 = (value: unknown): number => Math.max(0, Math.min(1, finite(value)));
const ZERO_CONTEXT_SCORES = normalizeEmotionScores({});

function scoresFrom(value: unknown): EmotionScores {
  if (!isRecord(value)) return normalizeEmotionScores({});
  const partial: Partial<Record<Emotion, number>> = {};
  for (const emotion of SHARED_EMOTIONS) {
    const score = value[emotion];
    if (typeof score === 'number' && Number.isFinite(score)) partial[emotion] = score;
  }
  return normalizeEmotionScores(partial);
}

function emotionFrom(value: unknown, scores: EmotionScores, inferWhenMissing = true): Emotion | null {
  if (value === null) return null;
  if (typeof value === 'string' && SHARED_EMOTIONS.includes(value as Emotion)) return value as Emotion;
  return inferWhenMissing ? dominantEmotion(scores) : null;
}

async function errorMessage(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as unknown;
  if (isRecord(body)) {
    for (const key of ['detail', 'message', 'error']) {
      if (typeof body[key] === 'string' && body[key]) return body[key];
    }
  }
  return `Request failed (${response.status})`;
}

export async function analyzeFace(jpeg: Blob, signal?: AbortSignal): Promise<FaceAnalysis> {
  const response = await fetch('/api/face', {
    method: 'POST',
    headers: { 'Content-Type': 'image/jpeg' },
    body: jpeg,
    signal,
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);

  const body = (await response.json()) as unknown;
  if (!isRecord(body) || typeof body.detected !== 'boolean') {
    throw new ApiError('Face service returned an invalid response.', response.status);
  }
  const scores = scoresFrom(body.scores);
  return {
    detected: body.detected,
    scores,
    retainedMass: clamp01(body.retained_mass),
    confidence: clamp01(body.confidence),
    dominant: body.detected ? emotionFrom(body.dominant, scores) : null,
    detail: typeof body.detail === 'string' ? body.detail : undefined,
  };
}

function parsePhrase(value: unknown): ConversationPhrase | null {
  if (!isRecord(value) || typeof value.text !== 'string') return null;
  const scores = scoresFrom(value.scores);
  return {
    text: value.text,
    emotion: emotionFrom(value.emotion, scores, false),
    scores,
    intensity: clamp01(value.intensity),
    evidence: Math.max(0, finite(value.evidence)),
    sharedMass: clamp01(
      value.shared_mass ?? SHARED_EMOTIONS.reduce((sum, emotion) => sum + scores[emotion], 0)
    ),
    direction: typeof value.direction === 'string' ? value.direction : '',
    startSeconds: typeof value.start_seconds === 'number' ? value.start_seconds : null,
    endSeconds: typeof value.end_seconds === 'number' ? value.end_seconds : null,
  };
}

function parseResponseContext(value: unknown): ResponseContext | null {
  if (!isRecord(value)) return null;
  const effects: ResponseContext['effect'][] = [
    'words-only', 'reinforced', 'adjusted', 'shifted', 'nonverbal-only', 'language-mixed', 'mixed',
  ];
  const strategies: ResponseContext['strategy'][] = [
    'celebrate', 'support', 'de-escalate', 'reassure', 'orient', 'stay-curious',
  ];
  if (
    typeof value.effect !== 'string'
    || !effects.includes(value.effect as ResponseContext['effect'])
    || typeof value.strategy !== 'string'
    || !strategies.includes(value.strategy as ResponseContext['strategy'])
    || typeof value.confidence !== 'number'
    || typeof value.nonverbal_weight !== 'number'
  ) return null;
  const sources = Array.isArray(value.sources)
    ? value.sources.filter((source): source is 'face' | 'prosody' => source === 'face' || source === 'prosody')
    : [];
  const sourceDominants: ResponseContext['sourceDominants'] = {};
  if (isRecord(value.source_dominants)) {
    for (const source of ['face', 'prosody'] as const) {
      if (source in value.source_dominants) {
        sourceDominants[source] = emotionFrom(value.source_dominants[source], ZERO_CONTEXT_SCORES, false);
      }
    }
  }
  return {
    dominant: emotionFrom(value.dominant, ZERO_CONTEXT_SCORES, false),
    confidence: clamp01(value.confidence),
    nonverbalWeight: clamp01(value.nonverbal_weight),
    nonverbalShift: typeof value.nonverbal_shift === 'number' ? clamp01(value.nonverbal_shift) : null,
    nonverbalDominant: emotionFrom(value.nonverbal_dominant, ZERO_CONTEXT_SCORES, false),
    languageDominant: emotionFrom(value.language_dominant, ZERO_CONTEXT_SCORES, false),
    effect: value.effect as ResponseContext['effect'],
    sources,
    sourceDominants,
    strategy: value.strategy as ResponseContext['strategy'],
  };
}

export async function converse(
  payload: ConversationRequest,
  signal?: AbortSignal
): Promise<ConversationResponse> {
  const response = await fetch('/api/conversation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);

  const body = (await response.json()) as unknown;
  if (!isRecord(body) || typeof body.transcript !== 'string' || typeof body.response !== 'string') {
    throw new ApiError('Conversation service returned an invalid response.', response.status);
  }

  const human = isRecord(body.human) ? body.human : {};
  const humanScores = scoresFrom(human.scores);
  const modalities: Record<string, ConversationModality> = {};
  if (isRecord(human.modalities)) {
    for (const [name, value] of Object.entries(human.modalities)) {
      if (!isRecord(value)) continue;
      modalities[name] = {
        scores: scoresFrom(value.scores),
        confidence: clamp01(value.confidence),
        available: typeof value.available === 'boolean' ? value.available : true,
      };
    }
  }
  const timings: Record<string, number> = {};
  if (isRecord(body.timings)) {
    for (const [key, value] of Object.entries(body.timings)) {
      if (typeof value === 'number' && Number.isFinite(value)) timings[key] = value;
    }
  }

  return {
    transcript: body.transcript,
    human: {
      scores: humanScores,
      dominant: emotionFrom(human.dominant, humanScores),
      confidence: clamp01(human.confidence),
      modalities,
    },
    responseContext: parseResponseContext(body.response_context),
    response: body.response,
    speechId: typeof body.speech_id === 'string' && body.speech_id ? body.speech_id : null,
    phrases: Array.isArray(body.phrases)
      ? body.phrases.map(parsePhrase).filter((phrase): phrase is ConversationPhrase => phrase !== null)
      : [],
    timings,
  };
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error('Could not read recorded audio.'));
    reader.onload = () => {
      const value = typeof reader.result === 'string' ? reader.result : '';
      const comma = value.indexOf(',');
      if (comma < 0) reject(new Error('Could not encode recorded audio.'));
      else resolve(value.slice(comma + 1));
    };
    reader.readAsDataURL(blob);
  });
}
