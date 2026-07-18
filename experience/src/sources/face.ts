import {
  clearEmotionModality,
  setEmotionModality,
  type EmotionScores,
} from '../state/emotion.js';
import { analyzeFace, type FaceAnalysis } from './api.js';

export type FaceSourceState = 'waiting' | 'live' | 'no-face' | 'unavailable';

export interface FaceSourceUpdate {
  state: FaceSourceState;
  scores: EmotionScores | null;
  confidence: number;
  dominant: FaceAnalysis['dominant'];
  detail?: string;
}

interface FaceSourceOptions {
  video: HTMLVideoElement;
  onUpdate: (update: FaceSourceUpdate) => void;
  intervalMs?: number;
}

const emptyUpdate = (state: FaceSourceState, detail?: string): FaceSourceUpdate => ({
  state,
  scores: null,
  confidence: 0,
  dominant: null,
  detail,
});

function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.72));
}

/**
 * Sample a mirrored preview without mirroring the bytes sent to the face model.
 * Requests never overlap; a slow model naturally lowers the capture rate.
 */
export function startFaceSource({
  video,
  onUpdate,
  intervalMs = 225,
}: FaceSourceOptions): () => void {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { alpha: false });
  let stopped = false;
  let timer = 0;
  let request: AbortController | null = null;
  let consecutiveFailures = 0;
  let hasSuccessfulResponse = false;

  onUpdate(emptyUpdate('waiting'));

  const schedule = (delay: number) => {
    if (!stopped) timer = window.setTimeout(() => void sample(), delay);
  };

  const sample = async () => {
    if (stopped) return;
    if (!context || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      schedule(intervalMs);
      return;
    }

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!sourceWidth || !sourceHeight) {
      schedule(intervalMs);
      return;
    }

    const width = Math.min(384, sourceWidth);
    const height = Math.max(1, Math.round((sourceHeight / sourceWidth) * width));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.drawImage(video, 0, 0, width, height);
    const jpeg = await canvasBlob(canvas);
    if (!jpeg || stopped) {
      schedule(intervalMs);
      return;
    }

    request = new AbortController();
    // HSEmotion may need to download and initialize on the first call. Keep that
    // first request singular and patient; only use a tighter timeout after the
    // service has proven warm.
    const timeout = window.setTimeout(
      () => request?.abort(),
      hasSuccessfulResponse ? 4200 : 20_000
    );
    try {
      const result = await analyzeFace(jpeg, request.signal);
      hasSuccessfulResponse = true;
      consecutiveFailures = 0;
      if (result.detected && result.dominant) {
        setEmotionModality('user', 'face', result.scores, result.confidence);
        onUpdate({
          state: 'live',
          scores: result.scores,
          confidence: result.confidence,
          dominant: result.dominant,
          detail: result.detail,
        });
      } else {
        clearEmotionModality('user', 'face');
        onUpdate(emptyUpdate('no-face', result.detail ?? 'Looking for one visible face.'));
      }
      schedule(intervalMs);
    } catch (error) {
      if (stopped) return;
      consecutiveFailures += 1;
      clearEmotionModality('user', 'face');
      const detail = error instanceof Error ? error.message : 'Face analysis is unavailable.';
      onUpdate(emptyUpdate('unavailable', detail));
      // Keep retrying, but do not hammer an unavailable local service.
      const aborted = error instanceof DOMException && error.name === 'AbortError';
      schedule(aborted && !hasSuccessfulResponse ? 10_000 : Math.min(5000, 900 * consecutiveFailures));
    } finally {
      window.clearTimeout(timeout);
      request = null;
    }
  };

  schedule(0);
  return () => {
    stopped = true;
    window.clearTimeout(timer);
    request?.abort();
    clearEmotionModality('user', 'face');
  };
}
