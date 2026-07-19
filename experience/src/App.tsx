import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { CrowdPane } from './crowd/CrowdPane.js';
import {
  SHARED_EMOTIONS,
  clearDirectEmotion,
  dominantEmotion,
  emitFrame,
  fuse,
  normalizeEmotionScores,
  setDirectEmotion,
  store,
  type Emotion,
  type EmotionScores,
} from './state/emotion.js';
import {
  blobToBase64,
  converse,
  type ConditioningMode,
  type ConversationPhrase,
  type ConversationRequest,
  type ConversationResponse,
} from './sources/api.js';
import {
  DEFAULT_DEMO_EMOTION,
  DEMO_PROOF_LINE,
  demoSignalSnapshot,
  startDemoSignals,
  type DemoSignalUpdate,
} from './sources/demo.js';
import { startFaceSource, type FaceSourceUpdate } from './sources/face.js';
import {
  startProsodySource,
  type ProsodyHandle,
  type ProsodySnapshot,
} from './sources/prosody.js';

type Phase =
  | 'idle'
  | 'requesting'
  | 'ready'
  | 'listening'
  | 'transcribing'
  | 'thinking'
  | 'awaiting-audio'
  | 'speaking'
  | 'holding'
  | 'complete'
  | 'error';

type SignalStatus = 'off' | 'waiting' | 'live' | 'no-face' | 'unavailable' | 'demo' | 'withheld';

interface SignalView {
  status: SignalStatus;
  scores: EmotionScores;
  confidence: number;
  dominant: Emotion | null;
  detail?: string;
}

interface CrowdView {
  scores: EmotionScores;
  confidence: number;
  dominant: Emotion | null;
  active: boolean;
}

interface FaceCaptureAccumulator {
  active: boolean;
  frames: number;
  detectedFrames: number;
  weight: number;
  confidence: number;
  scores: EmotionScores;
}

interface CounterfactualProof {
  transcript: string;
  emotion: Emotion;
  baseline: ConversationResponse;
  adapted: ConversationResponse | null;
}

type ProofStage = 'baseline' | 'adapted' | 'failed' | null;

const ZERO_SCORES = normalizeEmotionScores({});
const UNIFORM_SCORES = normalizeEmotionScores(
  Object.fromEntries(SHARED_EMOTIONS.map((emotion) => [emotion, 0.2]))
);

const emptyFaceCapture = (active = false): FaceCaptureAccumulator => ({
  active,
  frames: 0,
  detectedFrames: 0,
  weight: 0,
  confidence: 0,
  scores: normalizeEmotionScores({}),
});

const EMOTION_LABELS: Record<Emotion, string> = {
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
  afraid: 'Afraid',
  surprised: 'Surprised',
};

const PHASE_COPY: Record<Phase, string> = {
  idle: 'Your camera and microphone stay off until you begin.',
  requesting: 'Waiting for camera and microphone permission…',
  ready: 'Ready. Speak naturally, then stop when your thought is complete.',
  listening: 'Listening — face and voice energy are moving the left crowd.',
  transcribing: 'ElevenLabs Scribe and Gemma are processing through the local backend; no intermediate progress is inferred.',
  thinking: 'Gemma is forming a response and tracing its internal emotion vectors…',
  'awaiting-audio': 'The response is ready. Press play to hear the expressive voice.',
  speaking: 'Speaking — every word and the right crowd share one overall response emotion.',
  holding: 'Holding the overall response emotion for a moment.',
  complete: 'Response complete. The mirror is ready for another turn.',
  error: 'The last step could not complete. Nothing has been fabricated.',
};

const MAX_RECORDING_SECONDS = 45;
const MODEL_RESULT_HOLD_MS = 5 * 60 * 1000;

function confidenceLabel(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function resetAudioElement(audio: HTMLAudioElement | null): void {
  if (!audio) return;
  audio.pause();
  try {
    audio.currentTime = 0;
  } catch {
    // A stream without seek metadata can still be discarded safely.
  }
  if (audio.getAttribute('src')) {
    audio.removeAttribute('src');
    audio.load();
  }
}

function strongestEmotionalPhrase(
  phrases: ConversationPhrase[]
): { phrase: ConversationPhrase; index: number } | null {
  let strongest: { phrase: ConversationPhrase; index: number } | null = null;
  phrases.forEach((phrase, index) => {
    if (!phrase.emotion || phrase.intensity <= 0) return;
    if (!strongest || phrase.intensity > strongest.phrase.intensity) {
      strongest = { phrase, index };
    }
  });
  return strongest;
}

function showModelPhrase(phrase: ConversationPhrase | null | undefined): void {
  if (!phrase?.emotion) {
    clearDirectEmotion('model');
    return;
  }
  // Phrase scores already preserve the all-nine mass outside the shared five.
  // Use one authoritative categorical reading; intensity controls energy and
  // presentation separately so it cannot attenuate the population twice.
  // Keep the final measured tone visible while a judge discusses the result.
  // Every new turn and preset change clears it explicitly.
  setDirectEmotion('model', phrase.scores, 1, MODEL_RESULT_HOLD_MS);
}

function emitModelExpression(phrase: ConversationPhrase): void {
  emitFrame({
    source: 'gemma-vector-intensity',
    side: 'model',
    at: performance.now(),
    confidence: 1,
    channels: {
      // crowd.ts maps these back to 0..1. No valence or effort is invented.
      intensity: phrase.intensity * 3 - 1,
      movement: phrase.intensity * 2,
    },
  });
}

function mediaRecorderMime(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4;codecs=mp4a.40.2',
    'audio/mp4',
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

function currentCrowdView(side: 'user' | 'model'): CrowdView {
  const emotion = store[side].emotion;
  return {
    scores: { ...emotion.scores },
    confidence: emotion.confidence,
    dominant: emotion.active ? dominantEmotion(emotion.scores) : null,
    active: emotion.active,
  };
}

function faceView(update: FaceSourceUpdate): SignalView {
  return {
    status: update.state,
    scores: update.scores ?? ZERO_SCORES,
    confidence: update.confidence,
    dominant: update.dominant,
    detail: update.detail,
  };
}

function prosodyView(snapshot: ProsodySnapshot): SignalView {
  return {
    status: snapshot.voiced ? 'live' : 'waiting',
    scores: snapshot.scores,
    confidence: snapshot.confidence,
    dominant: snapshot.voiced ? dominantEmotion(snapshot.scores) : null,
    detail: snapshot.voiced
      ? `${Math.round(snapshot.pitchHz ?? 0)} Hz · ${Math.round(snapshot.spectralCentroidHz)} Hz centroid`
      : 'Waiting for voiced audio.',
  };
}

function EmotionHistogram({ scores, active }: { scores: EmotionScores; active: boolean }) {
  const total = SHARED_EMOTIONS.reduce((sum, emotion) => sum + scores[emotion], 0);
  const values = SHARED_EMOTIONS.map((emotion) => ({
    emotion,
    percent: total > 0 ? Math.round(scores[emotion] / total * 100) : 0,
  }));
  const description = values
    .map(({ emotion, percent }) => `${EMOTION_LABELS[emotion]} ${percent}%`)
    .join(', ');

  return (
    <div
      className={`emotion-bars${active ? '' : ' is-muted'}`}
      role="img"
      aria-label={`Relative emotion distribution: ${description}`}
    >
      {values.map(({ emotion, percent }) => (
        <span
          className={`emotion-meter emotion-meter--${emotion}`}
          key={emotion}
          title={`${EMOTION_LABELS[emotion]} ${percent}%`}
          aria-hidden="true"
        >
          <small>{EMOTION_LABELS[emotion]}</small>
          <em>{percent}%</em>
          <i><b style={{ height: `${percent}%` }} /></i>
        </span>
      ))}
    </div>
  );
}

function SignalChip({ label, signal, caveat }: { label: string; signal: SignalView; caveat?: string }) {
  const status = signal.status === 'live' || signal.status === 'demo'
    ? `${signal.dominant ? EMOTION_LABELS[signal.dominant] : 'mixed'} · ${confidenceLabel(signal.confidence)}`
    : signal.status === 'withheld'
      ? 'withheld for control'
    : signal.status === 'no-face'
      ? 'no face'
      : signal.status === 'unavailable'
        ? 'unavailable'
        : 'waiting';
  return (
    <div className={`signal-chip signal-chip--${signal.status}`} title={signal.detail}>
      <span className="signal-dot" aria-hidden="true" />
      <div>
        <strong>{label}</strong>
        <span>{status}</span>
      </div>
      {caveat ? <small>{caveat}</small> : null}
    </div>
  );
}

function phraseForTime(
  phrases: ConversationPhrase[],
  currentTime: number,
  duration: number
): number {
  if (!phrases.length) return -1;
  let previousStart = -Infinity;
  const hasTiming = phrases.every((phrase) => {
    const start = phrase.startSeconds;
    if (start === null || !Number.isFinite(start) || start < 0 || start < previousStart) return false;
    previousStart = start;
    return true;
  });
  if (hasTiming) {
    let candidate = 0;
    for (let index = 0; index < phrases.length; index += 1) {
      const start = phrases[index].startSeconds ?? 0;
      if (start <= currentTime + 0.04) candidate = index;
      else break;
    }
    return candidate;
  }
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  const totalCharacters = phrases.reduce((sum, phrase) => sum + Math.max(1, phrase.text.length), 0);
  let boundary = 0;
  const progress = Math.max(0, Math.min(1, currentTime / duration));
  for (let index = 0; index < phrases.length; index += 1) {
    boundary += Math.max(1, phrases[index].text.length) / totalCharacters;
    if (progress <= boundary) return index;
  }
  return phrases.length - 1;
}

interface TimedVoiceWord {
  text: string;
  emotion: Emotion;
  intensity: number;
  start: number;
  end: number;
}

function strongestResponseEmotion(
  phrases: ConversationPhrase[],
  fallback: Emotion | null
): Emotion {
  const representative = strongestEmotionalPhrase(phrases)?.phrase.emotion;
  const aggregateScores = Object.fromEntries(
    SHARED_EMOTIONS.map((emotion) => [
      emotion,
      phrases.reduce((sum, phrase) => sum + phrase.scores[emotion], 0),
    ])
  ) as EmotionScores;
  // The ribbon is deliberately one forced-choice reading for the complete
  // response. Phrase timing changes the active word, never its emotion color.
  return representative
    ?? dominantEmotion(aggregateScores)
    ?? fallback
    ?? SHARED_EMOTIONS[0];
}

function timedVoiceWords(
  phrases: ConversationPhrase[],
  response: string,
  duration: number,
  fallbackEmotion: Emotion | null
): TimedVoiceWord[] {
  const responseEmotion = strongestResponseEmotion(phrases, fallbackEmotion);
  const sourcePhrases = phrases.length
    ? phrases
    : [{
        text: response,
        emotion: null,
        intensity: 0,
        startSeconds: 0,
        endSeconds: duration || Math.max(2, response.length / 14),
      }];
  const fallbackDuration = duration > 0
    ? duration
    : Math.max(2, sourcePhrases.reduce((sum, phrase) => sum + phrase.text.length, 0) / 14);
  const totalCharacters = sourcePhrases.reduce(
    (sum, phrase) => sum + Math.max(1, phrase.text.length),
    0
  );
  let characterCursor = 0;
  const result: TimedVoiceWord[] = [];

  sourcePhrases.forEach((phrase) => {
    const fallbackStart = fallbackDuration * characterCursor / totalCharacters;
    characterCursor += Math.max(1, phrase.text.length);
    const fallbackEnd = fallbackDuration * characterCursor / totalCharacters;
    const start = phrase.startSeconds !== null && Number.isFinite(phrase.startSeconds)
      ? phrase.startSeconds
      : fallbackStart;
    const end = phrase.endSeconds !== null
      && Number.isFinite(phrase.endSeconds)
      && phrase.endSeconds > start
      ? phrase.endSeconds
      : Math.max(start + 0.25, fallbackEnd);
    const words = phrase.text.match(/\S+/g) ?? [];
    const totalWeight = words.reduce((sum, word) => sum + Math.max(1, word.length), 0);
    let wordCursor = 0;
    words.forEach((word) => {
      const wordStart = start + (end - start) * wordCursor / Math.max(1, totalWeight);
      wordCursor += Math.max(1, word.length);
      const wordEnd = start + (end - start) * wordCursor / Math.max(1, totalWeight);
      result.push({
        text: word,
        emotion: responseEmotion,
        intensity: phrase.intensity,
        start: wordStart,
        end: Math.max(wordStart + 0.06, wordEnd),
      });
    });
  });
  return result;
}

function ModelVoice({
  conversation,
  playbackTime,
  playbackDuration,
  phase,
  dominant,
  strength,
  needsPlay,
  onPlay,
}: {
  conversation: ConversationResponse;
  playbackTime: number;
  playbackDuration: number;
  phase: Phase;
  dominant: Emotion | null;
  strength: number;
  needsPlay: boolean;
  onPlay: () => void;
}) {
  const words = useMemo(
    () => timedVoiceWords(
      conversation.phrases,
      conversation.response,
      playbackDuration,
      dominant
    ),
    [conversation, playbackDuration, dominant]
  );
  const speaking = phase === 'speaking';
  const finished = phase === 'holding' || phase === 'complete';
  return (
    <section className="model-voice" aria-label="Gemma response synchronized to measured emotion">
      <header>
        <span>
          GEMMA VOICE · {conversation.conditioning.mode === 'vector' ? 'VECTOR L28' : 'PROMPT'}
        </span>
        <strong>{dominant ? `${EMOTION_LABELS[dominant]} · ${confidenceLabel(strength)}` : 'measured response'}</strong>
      </header>
      <p>
        {words.map((word, index) => {
          const active = speaking && playbackTime >= word.start && playbackTime < word.end;
          const spoken = playbackTime >= word.end || finished;
          return (
            <span
              key={`${index}-${word.text}`}
              className={`model-word${active ? ' is-active' : ''}${spoken ? ' is-spoken' : ''}`}
              style={{
                '--word-emotion': `var(--${word.emotion})`,
                '--word-strength': `${Math.round(word.intensity * 100)}%`,
              } as CSSProperties}
            >
              {word.text}
            </span>
          );
        })}
      </p>
      {(needsPlay || finished) && conversation.speechId ? (
        <button type="button" className="voice-play" onClick={onPlay}>
          {needsPlay ? '▶ Play expressive voice' : '↻ Replay voice'}
        </button>
      ) : null}
    </section>
  );
}

export function App() {
  const demoRequested = useMemo(
    () => new URLSearchParams(window.location.search).get('demo') === '1',
    []
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef(0);
  const recordingLimitRef = useRef(0);
  const faceCaptureRef = useRef<FaceCaptureAccumulator>(emptyFaceCapture());
  const faceStopRef = useRef<(() => void) | null>(null);
  const prosodyRef = useRef<ProsodyHandle | null>(null);
  const demoFaceStopRef = useRef<(() => void) | null>(null);
  const demoProsodyStopRef = useRef<(() => void) | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const demoEmotionRef = useRef<Emotion>(DEFAULT_DEMO_EMOTION);
  const judgeProofActiveRef = useRef(false);
  const latestFaceRef = useRef<SignalView>({
    status: 'off', scores: ZERO_SCORES, confidence: 0, dominant: null,
  });
  const latestProsodyRef = useRef<SignalView>({
    status: 'off', scores: UNIFORM_SCORES, confidence: 0, dominant: null,
  });
  const activePhraseRef = useRef(-1);

  const [phase, setPhase] = useState<Phase>('idle');
  const [started, setStarted] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recorderAvailable, setRecorderAvailable] = useState(true);
  const [permissionNote, setPermissionNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [typedText, setTypedText] = useState('');
  const [conversation, setConversation] = useState<ConversationResponse | null>(null);
  const [faceSignal, setFaceSignal] = useState<SignalView>(latestFaceRef.current);
  const [prosodySignal, setProsodySignal] = useState<SignalView>(latestProsodyRef.current);
  const [userCrowd, setUserCrowd] = useState<CrowdView>(() => currentCrowdView('user'));
  const [modelCrowd, setModelCrowd] = useState<CrowdView>(() => currentCrowdView('model'));
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  const [needsPlay, setNeedsPlay] = useState(false);
  const [voiceNotice, setVoiceNotice] = useState<string | null>(null);
  const [demoFaceActive, setDemoFaceActive] = useState(false);
  const [demoProsodyActive, setDemoProsodyActive] = useState(false);
  const [demoEmotion, setDemoEmotion] = useState<Emotion>(DEFAULT_DEMO_EMOTION);
  const [judgeProofActive, setJudgeProofActive] = useState(false);
  const [counterfactualProof, setCounterfactualProof] = useState<CounterfactualProof | null>(null);
  const [proofStage, setProofStage] = useState<ProofStage>(null);
  const [conditioningMode, setConditioningMode] = useState<ConditioningMode>('prompt');

  useEffect(() => {
    let frame = 0;
    let lastUiUpdate = 0;
    const tick = (now: number) => {
      fuse();
      if (now - lastUiUpdate > 140) {
        lastUiUpdate = now;
        setUserCrowd(currentCrowdView('user'));
        setModelCrowd(currentCrowdView('model'));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const recordFaceFrame = useCallback((signal: SignalView) => {
    const capture = faceCaptureRef.current;
    if (!capture.active) return;
    capture.frames += 1;
    if (signal.confidence <= 0) return;
    capture.detectedFrames += 1;
    capture.confidence += signal.confidence;
    capture.weight += signal.confidence;
    for (const emotion of SHARED_EMOTIONS) {
      capture.scores[emotion] += signal.scores[emotion] * signal.confidence;
    }
  }, []);

  const finishFaceCapture = useCallback((): { scores: EmotionScores; confidence: number } | null => {
    const capture = faceCaptureRef.current;
    capture.active = false;
    if (!capture.detectedFrames || !capture.weight) return null;
    const average: Partial<EmotionScores> = {};
    for (const emotion of SHARED_EMOTIONS) {
      average[emotion] = capture.scores[emotion] / capture.weight;
    }
    const coverage = capture.detectedFrames / Math.max(1, capture.frames);
    return {
      scores: normalizeEmotionScores(average),
      confidence: Math.min(
        1,
        (capture.confidence / capture.detectedFrames) * Math.sqrt(coverage)
      ),
    };
  }, []);

  const updateDemoSignal = useCallback((update: DemoSignalUpdate, source: 'face' | 'prosody') => {
    if (source === 'face') {
      const signal: SignalView = {
        status: 'demo',
        scores: update.face,
        confidence: update.faceConfidence,
        dominant: dominantEmotion(update.face),
        detail: `${EMOTION_LABELS[demoEmotionRef.current]} injected rehearsal signal; not a camera inference.`,
      };
      recordFaceFrame(signal);
      latestFaceRef.current = signal;
      setFaceSignal(signal);
    } else {
      const signal: SignalView = {
        status: 'demo',
        scores: update.prosody,
        confidence: update.prosodyConfidence,
        dominant: dominantEmotion(update.prosody),
        detail: `${EMOTION_LABELS[demoEmotionRef.current]} injected rehearsal signal; not microphone analysis.`,
      };
      latestProsodyRef.current = signal;
      setProsodySignal(signal);
    }
  }, [recordFaceFrame]);

  const startDemoFace = useCallback(() => {
    if (!demoRequested || demoFaceStopRef.current) return;
    demoFaceStopRef.current = startDemoSignals(
      (update) => updateDemoSignal(update, 'face'),
      {
        face: true,
        prosody: false,
        emotion: () => demoEmotionRef.current,
      }
    );
    setDemoFaceActive(true);
  }, [demoRequested, updateDemoSignal]);

  const startDemoProsody = useCallback(() => {
    if (!demoRequested || demoProsodyStopRef.current) return;
    demoProsodyStopRef.current = startDemoSignals(
      (update) => updateDemoSignal(update, 'prosody'),
      {
        face: false,
        prosody: true,
        emotion: () => demoEmotionRef.current,
      }
    );
    setDemoProsodyActive(true);
  }, [demoRequested, updateDemoSignal]);

  const withholdDemoSignals = useCallback(() => {
    demoFaceStopRef.current?.();
    demoFaceStopRef.current = null;
    demoProsodyStopRef.current?.();
    demoProsodyStopRef.current = null;
    setDemoFaceActive(false);
    setDemoProsodyActive(false);
    const withheldFace: SignalView = {
      status: 'withheld',
      scores: ZERO_SCORES,
      confidence: 0,
      dominant: null,
      detail: 'Deliberately withheld from the transcript-only control.',
    };
    const withheldProsody: SignalView = {
      ...withheldFace,
      scores: UNIFORM_SCORES,
    };
    latestFaceRef.current = withheldFace;
    latestProsodyRef.current = withheldProsody;
    setFaceSignal(withheldFace);
    setProsodySignal(withheldProsody);
    setPermissionNote('Control run: the selected face and voice context are deliberately withheld.');
    clearDirectEmotion('user');
  }, []);

  const onFaceUpdate = useCallback((update: FaceSourceUpdate) => {
    if (update.state !== 'unavailable' && demoFaceStopRef.current) {
      demoFaceStopRef.current();
      demoFaceStopRef.current = null;
      setDemoFaceActive(false);
    }
    if (update.state === 'unavailable' && demoRequested) {
      startDemoFace();
      return;
    }
    const signal = faceView(update);
    recordFaceFrame(signal);
    latestFaceRef.current = signal;
    setFaceSignal(signal);
  }, [demoRequested, recordFaceFrame, startDemoFace]);

  const onProsodyUpdate = useCallback((snapshot: ProsodySnapshot) => {
    const signal = prosodyView(snapshot);
    latestProsodyRef.current = signal;
    setProsodySignal(signal);
  }, []);

  const startExperience = useCallback(async () => {
    judgeProofActiveRef.current = false;
    setJudgeProofActive(false);
    setPhase('requesting');
    setError(null);
    setPermissionNote(null);
    let acquiredStream: MediaStream | null = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no media capture support.');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      acquiredStream = stream;
      mediaRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
        faceStopRef.current = startFaceSource({ video, onUpdate: onFaceUpdate });
      }
      try {
        prosodyRef.current = await startProsodySource(stream, onProsodyUpdate);
      } catch (prosodyError) {
        const message = prosodyError instanceof Error ? prosodyError.message : 'Prosody analysis is unavailable.';
        const unavailable: SignalView = {
          status: 'unavailable', scores: UNIFORM_SCORES, confidence: 0, dominant: null, detail: message,
        };
        latestProsodyRef.current = unavailable;
        setProsodySignal(unavailable);
        startDemoProsody();
      }
      setRecorderAvailable(typeof MediaRecorder !== 'undefined' && stream.getAudioTracks().length > 0);
      setStarted(true);
      setPhase('ready');
    } catch (mediaError) {
      faceStopRef.current?.();
      faceStopRef.current = null;
      prosodyRef.current?.stop();
      prosodyRef.current = null;
      acquiredStream?.getTracks().forEach((track) => track.stop());
      if (mediaRef.current === acquiredStream) mediaRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      const message = mediaError instanceof Error ? mediaError.message : 'Camera and microphone permission failed.';
      setPermissionNote(message);
      setStarted(true);
      setRecorderAvailable(false);
      const unavailable: SignalView = {
        status: 'unavailable', scores: ZERO_SCORES, confidence: 0, dominant: null, detail: message,
      };
      latestFaceRef.current = unavailable;
      latestProsodyRef.current = { ...unavailable, scores: UNIFORM_SCORES };
      setFaceSignal(unavailable);
      setProsodySignal({ ...unavailable, scores: UNIFORM_SCORES });
      if (demoRequested) {
        startDemoFace();
        startDemoProsody();
        setPhase('ready');
      } else {
        setPhase('error');
      }
    }
  }, [demoRequested, onFaceUpdate, onProsodyUpdate, startDemoFace, startDemoProsody]);

  const continueTyped = useCallback(() => {
    setStarted(true);
    setRecorderAvailable(false);
    setPermissionNote('Camera and microphone were not started. Typed text remains available.');
    setPhase('ready');
    if (demoRequested) {
      startDemoFace();
      startDemoProsody();
    }
  }, [demoRequested, startDemoFace, startDemoProsody]);

  const chooseDemoEmotion = useCallback((emotion: Emotion) => {
    demoEmotionRef.current = emotion;
    setDemoEmotion(emotion);
    const snapshot = demoSignalSnapshot(emotion);
    updateDemoSignal(snapshot, 'face');
    updateDemoSignal(snapshot, 'prosody');
    resetAudioElement(audioRef.current);
    clearDirectEmotion('user');
    clearDirectEmotion('model');
    activePhraseRef.current = -1;
    setConversation(null);
    setCounterfactualProof(null);
    setProofStage(null);
    setTypedText(DEMO_PROOF_LINE);
    setNeedsPlay(false);
    setVoiceNotice(null);
    setError(null);
    setPermissionNote(
      `Controlled rehearsal mode: labeled ${EMOTION_LABELS[emotion].toLowerCase()} face and voice test signals are injected.`
    );
    setPhase('ready');
  }, [updateDemoSignal]);

  const openJudgeProof = useCallback(() => {
    judgeProofActiveRef.current = true;
    setJudgeProofActive(true);
    chooseDemoEmotion(DEFAULT_DEMO_EMOTION);
    setStarted(true);
    setRecorderAvailable(false);
    setPermissionNote(
      'Controlled rehearsal mode: face and voice are injected test signals, permanently labeled below.'
    );
    startDemoFace();
    startDemoProsody();
  }, [chooseDemoEmotion, startDemoFace, startDemoProsody]);

  const signalsForRequest = useCallback((
    payload: ConversationRequest,
    prosodyOverride?: { scores: EmotionScores; confidence: number },
    faceOverride?: { scores: EmotionScores; confidence: number } | null
  ): ConversationRequest => {
    payload.conditioning_mode = conditioningMode;
    // `undefined` means a typed turn may use the current live frame. An
    // explicit `null` means a recorded take contained no detected face and
    // must not fall back to a stale pre-recording frame.
    const face = faceOverride === undefined ? latestFaceRef.current : faceOverride;
    const prosody = prosodyOverride ?? latestProsodyRef.current;
    if (face && face.confidence > 0) {
      payload.face_scores = face.scores;
      payload.face_confidence = face.confidence;
    }
    if (prosody.confidence > 0) {
      payload.prosody_scores = prosody.scores;
      payload.prosody_confidence = prosody.confidence;
    }
    return payload;
  }, [conditioningMode]);

  const presentConversation = useCallback((result: ConversationResponse) => {
    // The backend result is already the authoritative fusion of language,
    // face and utterance-averaged prosody. Keep it direct so raw modalities
    // are not counted a second time.
    if (judgeProofActiveRef.current) {
      // Rehearsal presets are stable, continuously refreshed sources. A
      // decaying turn snapshot would make the labeled preset look weaker
      // while judges discuss the result.
      clearDirectEmotion('user');
    } else {
      setDirectEmotion('user', result.human.scores, result.human.confidence);
    }
    setConversation(result);
    const representative = strongestEmotionalPhrase(result.phrases);
    showModelPhrase(representative?.phrase);
    if (result.speechId) {
      setPhase('speaking');
    } else {
      setVoiceNotice('Expressive voice is unavailable; the returned emotion trace is still shown.');
      setPhase('complete');
    }
  }, []);

  const submitConversation = useCallback(async (
    payload: ConversationRequest,
    startsWithTranscription: boolean
  ) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const requestId = ++requestIdRef.current;
    resetAudioElement(audioRef.current);
    clearDirectEmotion('user');
    clearDirectEmotion('model');
    activePhraseRef.current = -1;
    setConversation(null);
    setCounterfactualProof(null);
    setProofStage(null);
    setNeedsPlay(false);
    setVoiceNotice(null);
    setError(null);
    setPhase(startsWithTranscription ? 'transcribing' : 'thinking');
    try {
      const result = await converse(payload, controller.signal);
      if (requestIdRef.current !== requestId) return;
      presentConversation(result);
    } catch (requestError) {
      if (controller.signal.aborted) return;
      const message = requestError instanceof Error ? requestError.message : 'Conversation request failed.';
      setError(message);
      setPhase('error');
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
    }
  }, [presentConversation]);

  const submitRecording = useCallback(async (
    blob: Blob,
    average: { scores: EmotionScores; confidence: number },
    faceAverage: { scores: EmotionScores; confidence: number } | null
  ) => {
    try {
      const audioBase64 = await blobToBase64(blob);
      const payload = signalsForRequest({
        audio_base64: audioBase64,
        audio_content_type: blob.type || 'audio/webm',
      }, average, faceAverage);
      await submitConversation(payload, true);
    } catch (recordingError) {
      const message = recordingError instanceof Error ? recordingError.message : 'Recorded audio could not be sent.';
      setError(message);
      setPhase('error');
    }
  }, [signalsForRequest, submitConversation]);

  const startRecording = useCallback(() => {
    const stream = mediaRef.current;
    if (!stream || !stream.getAudioTracks().length || typeof MediaRecorder === 'undefined') {
      setError('Live recording is unavailable. Use the typed fallback below.');
      setPhase('error');
      return;
    }
    const mimeType = mediaRecorderMime();
    resetAudioElement(audioRef.current);
    clearDirectEmotion('user');
    clearDirectEmotion('model');
    activePhraseRef.current = -1;
    const audioOnly = new MediaStream(stream.getAudioTracks());
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(audioOnly, mimeType ? { mimeType } : undefined);
    } catch (recorderError) {
      const message = recorderError instanceof Error ? recorderError.message : 'The browser could not initialize audio recording.';
      setError(message);
      setPhase('error');
      return;
    }
    recorderRef.current = recorder;
    recordingChunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) recordingChunksRef.current.push(event.data);
    };
    recorder.onerror = () => {
      window.clearInterval(recordingTimerRef.current);
      window.clearTimeout(recordingLimitRef.current);
      prosodyRef.current?.endCapture();
      faceCaptureRef.current.active = false;
      recorder.onstop = null;
      recorderRef.current = null;
      setRecording(false);
      setError('The browser could not record this utterance.');
      setPhase('error');
    };
    recorder.onstop = () => {
      window.clearInterval(recordingTimerRef.current);
      window.clearTimeout(recordingLimitRef.current);
      setRecording(false);
      const average = prosodyRef.current?.endCapture() ?? {
        scores: latestProsodyRef.current.scores,
        confidence: latestProsodyRef.current.confidence,
      };
      const faceAverage = finishFaceCapture();
      const blob = new Blob(recordingChunksRef.current, {
        type: recorder.mimeType || mimeType || 'audio/webm',
      });
      recordingChunksRef.current = [];
      recorderRef.current = null;
      if (!blob.size) {
        setError('No microphone audio was captured.');
        setPhase('error');
        return;
      }
      setPhase('transcribing');
      void submitRecording(blob, average, faceAverage);
    };
    try {
      faceCaptureRef.current = emptyFaceCapture(true);
      prosodyRef.current?.beginCapture();
      recorder.start(200);
    } catch (recordingError) {
      prosodyRef.current?.endCapture();
      faceCaptureRef.current.active = false;
      recorderRef.current = null;
      const message = recordingError instanceof Error
        ? recordingError.message
        : 'The browser could not start audio recording.';
      setError(message);
      setPhase('error');
      return;
    }
    const startedAt = performance.now();
    setRecordingSeconds(0);
    recordingTimerRef.current = window.setInterval(() => {
      setRecordingSeconds(Math.min(MAX_RECORDING_SECONDS, Math.floor((performance.now() - startedAt) / 1000)));
    }, 250);
    recordingLimitRef.current = window.setTimeout(() => {
      if (recorder.state === 'recording') recorder.stop();
    }, MAX_RECORDING_SECONDS * 1000);
    setConversation(null);
    setVoiceNotice(null);
    setError(null);
    setRecording(true);
    setPhase('listening');
  }, [finishFaceCapture, submitRecording]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder?.state === 'recording') recorder.stop();
  }, []);

  const submitTyped = useCallback(() => {
    if (
      recording || phase === 'requesting' || phase === 'transcribing' || phase === 'thinking' ||
      phase === 'speaking' || phase === 'holding' || phase === 'awaiting-audio'
    ) return;
    const transcript = typedText.trim();
    if (!transcript) return;
    setTypedText('');
    void submitConversation(signalsForRequest({ transcript }), false);
  }, [phase, recording, signalsForRequest, submitConversation, typedText]);

  const runCounterfactualProof = useCallback(async () => {
    const proofRunning = proofStage === 'baseline' || proofStage === 'adapted';
    if (!judgeProofActive || proofRunning) return;
    if (
      recording || phase === 'requesting' || phase === 'transcribing'
      || phase === 'thinking' || phase === 'speaking' || phase === 'holding'
      || phase === 'awaiting-audio'
    ) return;
    const transcript = typedText.trim() || DEMO_PROOF_LINE;
    const emotion = demoEmotionRef.current;
    const injected = demoSignalSnapshot(emotion);
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const requestId = ++requestIdRef.current;
    resetAudioElement(audioRef.current);
    clearDirectEmotion('user');
    clearDirectEmotion('model');
    activePhraseRef.current = -1;
    setConversation(null);
    setCounterfactualProof(null);
    setNeedsPlay(false);
    setVoiceNotice(null);
    setError(null);
    withholdDemoSignals();
    setProofStage('baseline');
    setPhase('thinking');
    let baseline: ConversationResponse | null = null;
    try {
      baseline = await converse(
        {
          transcript,
          synthesize_speech: false,
          conditioning_mode: conditioningMode,
        },
        controller.signal
      );
      if (requestIdRef.current !== requestId) return;
      setCounterfactualProof({ transcript, emotion, baseline, adapted: null });
      startDemoFace();
      startDemoProsody();
      setPermissionNote(
        `Adapted run: labeled ${EMOTION_LABELS[emotion].toLowerCase()} face and voice test signals are now injected.`
      );
      setProofStage('adapted');

      const adapted = await converse(
        {
          transcript,
          face_scores: injected.face,
          face_confidence: injected.faceConfidence,
          prosody_scores: injected.prosody,
          prosody_confidence: injected.prosodyConfidence,
          synthesize_speech: true,
          conditioning_mode: conditioningMode,
        },
        controller.signal
      );
      if (requestIdRef.current !== requestId) return;
      setCounterfactualProof({ transcript, emotion, baseline, adapted });
      presentConversation(adapted);
    } catch (proofError) {
      if (controller.signal.aborted) return;
      setError(
        baseline
          ? 'The transcript-only control is preserved, but the adapted half of the proof could not complete.'
          : proofError instanceof Error
            ? proofError.message
            : 'The counterfactual proof could not complete.'
      );
      if (baseline) setProofStage('failed');
      else {
        startDemoFace();
        startDemoProsody();
        setPermissionNote(
          'Controlled rehearsal mode: face and voice are injected test signals, permanently labeled below.'
        );
        setProofStage(null);
      }
      setPhase('error');
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
      setProofStage((current) => current === 'failed' ? current : null);
    }
  }, [
    judgeProofActive,
    conditioningMode,
    phase,
    presentConversation,
    proofStage,
    recording,
    startDemoFace,
    startDemoProsody,
    typedText,
    withholdDemoSignals,
  ]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    setPlaybackTime(0);
    setPlaybackDuration(0);
    if (!conversation?.speechId) {
      resetAudioElement(audio);
      return;
    }
    const audioSource = `/api/audio/${encodeURIComponent(conversation.speechId)}`;
    const representative = strongestEmotionalPhrase(conversation.phrases);
    let animationFrame = 0;
    let holdTimer = 0;
    let lastExpressionFrame = 0;
    let lastWordFrame = 0;
    let playbackFailed = false;
    const sync = () => {
      const index = phraseForTime(conversation.phrases, audio.currentTime, audio.duration);
      if (index >= 0 && index !== activePhraseRef.current) {
        activePhraseRef.current = index;
        // The entire reply keeps one overall emotion; phrase boundaries only
        // advance the synchronized words and motion intensity.
        showModelPhrase(representative?.phrase);
      }
      const phrase = conversation.phrases[index];
      const expressionPhrase = phrase ?? representative?.phrase;
      const now = performance.now();
      if (now - lastWordFrame >= 65) {
        setPlaybackTime(audio.currentTime);
        if (Number.isFinite(audio.duration) && audio.duration > 0) {
          setPlaybackDuration(audio.duration);
        }
        lastWordFrame = now;
      }
      if (expressionPhrase && now - lastExpressionFrame >= 180) {
        emitModelExpression(expressionPhrase);
        lastExpressionFrame = now;
      }
      if (!audio.paused && !audio.ended) animationFrame = requestAnimationFrame(sync);
    };
    const onPlay = () => {
      window.clearTimeout(holdTimer);
      holdTimer = 0;
      playbackFailed = false;
      setNeedsPlay(false);
      setPhase('speaking');
      cancelAnimationFrame(animationFrame);
      // Refresh even when resuming inside the same phrase after its store TTL.
      activePhraseRef.current = -1;
      sync();
    };
    const onPause = () => {
      cancelAnimationFrame(animationFrame);
      if (!audio.ended) setPhase('awaiting-audio');
    };
    const onEnded = () => {
      cancelAnimationFrame(animationFrame);
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        setPlaybackTime(audio.duration);
        setPlaybackDuration(audio.duration);
      }
      showModelPhrase(representative?.phrase);
      activePhraseRef.current = -1;
      setPhase('holding');
      holdTimer = window.setTimeout(() => {
        setPhase('complete');
      }, 2200);
    };
    const onError = () => {
      playbackFailed = true;
      cancelAnimationFrame(animationFrame);
      window.clearTimeout(holdTimer);
      holdTimer = 0;
      const hasMeasuredEmotion = Boolean(representative);
      showModelPhrase(representative?.phrase);
      activePhraseRef.current = -1;
      setVoiceNotice(
        hasMeasuredEmotion
          ? 'The MP3 could not be loaded. The response and measured phrase emotion remain visible.'
          : 'The MP3 could not be loaded. The response remains visible; no shared phrase emotion met the evidence threshold.'
      );
      setPhase('complete');
    };
    const onLoadedMetadata = () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        setPlaybackDuration(audio.duration);
      }
    };
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('error', onError);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.src = audioSource;
    audio.load();
    void audio.play().catch(() => {
      if (playbackFailed || audio.error) return;
      setNeedsPlay(true);
      setPhase('awaiting-audio');
    });
    return () => {
      cancelAnimationFrame(animationFrame);
      window.clearTimeout(holdTimer);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('error', onError);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      if (audio.getAttribute('src') === audioSource) resetAudioElement(audio);
    };
  }, [conversation]);

  const playResponse = useCallback(() => {
    void audioRef.current?.play().catch(() => setVoiceNotice('Browser playback is still blocked. Use the audio controls.'));
  }, []);

  useEffect(() => () => {
    requestIdRef.current += 1;
    window.clearInterval(recordingTimerRef.current);
    window.clearTimeout(recordingLimitRef.current);
    requestRef.current?.abort();
    faceCaptureRef.current.active = false;
    faceStopRef.current?.();
    prosodyRef.current?.stop();
    demoFaceStopRef.current?.();
    demoProsodyStopRef.current?.();
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = null;
      recorder.stop();
    }
    mediaRef.current?.getTracks().forEach((track) => track.stop());
    clearDirectEmotion('user');
    clearDirectEmotion('model');
  }, []);

  const busy = phase === 'transcribing' || phase === 'thinking' || phase === 'requesting';
  const textLocked = busy || recording || phase === 'speaking' || phase === 'holding' || phase === 'awaiting-audio';
  const representativeModelPhrase = useMemo(
    () => strongestEmotionalPhrase(conversation?.phrases ?? [])?.phrase ?? null,
    [conversation]
  );
  const displayedModelPhrase = modelCrowd.active ? representativeModelPhrase : null;
  const displayedModelEmotion = displayedModelPhrase
    ? displayedModelPhrase.emotion
    : modelCrowd.dominant;
  const displayedModelStrength = displayedModelPhrase
    ? displayedModelPhrase.intensity
    : modelCrowd.confidence;
  const displayedModelScores = displayedModelPhrase?.scores ?? modelCrowd.scores;
  const responseContext = conversation?.responseContext;
  const vectorConditioning = conversation?.conditioning.mode === 'vector';
  const proofBaselineContext = counterfactualProof?.baseline.responseContext;
  const proofAdaptedContext = counterfactualProof?.adapted?.responseContext;
  const proofUsesVector = counterfactualProof?.baseline.conditioning.mode === 'vector';
  const proofBaselinePlan = proofUsesVector
    ? `L28 ${(proofBaselineContext?.dominant ?? 'mixed').toUpperCase()} VECTOR`
    : proofBaselineContext?.strategy.replace('-', ' ').toUpperCase() ?? 'UNAVAILABLE';
  const proofAdaptedPlan = proofUsesVector && proofAdaptedContext
    ? `L28 ${(proofAdaptedContext.dominant ?? 'mixed').toUpperCase()} VECTOR`
    : proofAdaptedContext?.strategy.replace('-', ' ').toUpperCase() ?? (
    proofStage === 'adapted' ? 'RUNNING…' : proofStage === 'failed' ? 'FAILED' : 'UNAVAILABLE'
  );
  const proofPlanChanged = Boolean(
    counterfactualProof?.adapted
    && (
      proofUsesVector
        ? proofBaselinePlan !== proofAdaptedPlan
        : proofBaselineContext?.strategy
          && proofAdaptedContext?.strategy
          && proofBaselineContext.strategy !== proofAdaptedContext.strategy
    )
  );
  const proofReplyChanged = Boolean(
    counterfactualProof?.adapted
    && counterfactualProof.baseline.response.trim().replace(/\s+/g, ' ').toLowerCase()
      !== counterfactualProof.adapted.response.trim().replace(/\s+/g, ' ').toLowerCase()
  );
  const proofChangeLabel = proofUsesVector ? 'VECTOR CHANGED' : 'PLAN CHANGED';
  const proofOutcomeCopy = !counterfactualProof?.adapted
    ? proofStage === 'failed'
      ? 'ADAPTED RUN FAILED · CONTROL PRESERVED · RETRY'
      : 'CONTROL CAPTURED · ADDING NONVERBAL CONTEXT…'
    : proofPlanChanged && proofReplyChanged
      ? `${proofChangeLabel} · REPLY CHANGED`
      : proofPlanChanged
        ? proofChangeLabel
        : proofReplyChanged
          ? `REPLY CHANGED WITH THE SAME ${proofUsesVector ? 'VECTOR' : 'PLAN'}`
          : 'CONTEXT CONFIRMED THE SAME OUTCOME';
  const proofShiftCopy = proofAdaptedContext?.nonverbalShift !== null
    && proofAdaptedContext?.nonverbalShift !== undefined
    ? ` · ${confidenceLabel(proofAdaptedContext.nonverbalShift)} SIGNAL-DISTRIBUTION SHIFT`
    : '';
  const nonverbalSourceLabel = responseContext?.sources
    .map((source) => source === 'face' ? 'FACE' : 'VOICE')
    .join(' + ') ?? '';
  const sourceReadings = responseContext?.sources
    .map((source) => {
      const label = source === 'face' ? 'FACE' : 'VOICE';
      return `${label} READ ${(responseContext.sourceDominants[source] ?? 'MIXED').toUpperCase()}`;
    })
    .join(' · ') ?? '';
  const strategyLabel = vectorConditioning
    ? `LAYER 28 · ${conversation?.conditioning.targetTokens ?? 0} TOKENS`
    : responseContext?.strategy.replace('-', ' ').toUpperCase() ?? '';
  const adaptationCopy = !responseContext
    ? ''
    : vectorConditioning
      ? `${(responseContext.dominant ?? 'MIXED').toUpperCase()} FUSED DIRECTION · ADDED TO THE FINAL USER SENTENCE · NO EMOTION LABELS OR SCORES IN THE PROMPT`
    : responseContext.effect === 'words-only'
      ? `WORDS SET THE CONTEXT · RESPONSE PLAN: ${strategyLabel}`
      : responseContext.effect === 'reinforced'
        ? `WORDS + ${nonverbalSourceLabel} AGREED: ${(responseContext.dominant ?? 'UNCERTAIN').toUpperCase()} · RESPONSE PLAN: ${strategyLabel}`
        : responseContext.effect === 'shifted'
          ? `WORDS LEANED ${(responseContext.languageDominant ?? 'UNCERTAIN').toUpperCase()} · ${sourceReadings} · FUSED RESPONSE PLAN: ${strategyLabel}`
          : responseContext.effect === 'adjusted'
            ? `WORDS LEANED ${(responseContext.languageDominant ?? 'UNCERTAIN').toUpperCase()} · ${sourceReadings} · BALANCED RESPONSE PLAN: ${strategyLabel}`
            : responseContext.effect === 'nonverbal-only'
              ? `WORDS HAD NO USABLE AFFECT SIGNAL · ${sourceReadings} · RESPONSE PLAN: ${strategyLabel}`
              : responseContext.effect === 'language-mixed'
                ? `WORDS WERE EMOTIONALLY MIXED · ${sourceReadings} · FUSED RESPONSE PLAN: ${strategyLabel}`
                : `${sourceReadings} · CONFLICT-AWARE RESPONSE PLAN: ${strategyLabel}`;
  const contextDetailCopy = responseContext
    ? [
        `${(responseContext.dominant ?? 'uncertain').toUpperCase()} CONTEXT`,
        `${confidenceLabel(responseContext.confidence)} FUSED SIGNAL STRENGTH`,
        responseContext.nonverbalWeight > 0
          ? `${confidenceLabel(responseContext.nonverbalWeight)} NONVERBAL FUSION WEIGHT`
          : null,
        responseContext.nonverbalShift !== null
          && (
            responseContext.effect === 'shifted'
            || responseContext.effect === 'adjusted'
            || responseContext.effect === 'language-mixed'
          )
          ? `${confidenceLabel(responseContext.nonverbalShift)} DISTRIBUTION SHIFT`
          : null,
        vectorConditioning
          ? `${confidenceLabel(conversation?.conditioning.appliedResidualRatio ?? 0)} RESIDUAL EDIT`
          : null,
      ].filter(Boolean).join(' · ')
    : '';
  const contextAxisValue = responseContext
    ? responseContext.effect === 'shifted'
      || responseContext.effect === 'adjusted'
      || responseContext.effect === 'language-mixed'
      ? responseContext.nonverbalShift ?? responseContext.nonverbalWeight
      : responseContext.effect === 'words-only'
        ? 0
        : responseContext.nonverbalWeight
    : 0;
  const contextAxisCopy = !responseContext
    ? conversation ? 'CONTEXT UNAVAILABLE' : 'AWAITING RESPONSE'
    : responseContext.effect === 'words-only'
      ? 'WORDS ONLY'
      : responseContext.effect === 'reinforced'
        ? 'SENSORS CONFIRMED'
        : responseContext.effect === 'mixed'
          ? 'SIGNALS MIXED'
          : responseContext.effect === 'nonverbal-only'
            ? 'SENSORS PRIMARY'
            : responseContext.effect === 'language-mixed'
              ? `WORDS MIXED · SHIFT ${confidenceLabel(responseContext.nonverbalShift ?? 0)}`
            : `SIGNAL SHIFT ${confidenceLabel(responseContext.nonverbalShift ?? 0)}`;
  const languageModality = conversation?.human.modalities.language;
  const languageSignal: SignalView | null = languageModality?.scores && languageModality.confidence !== undefined
    ? {
        status: languageModality.available === false ? 'unavailable' : 'live',
        scores: languageModality.scores,
        confidence: languageModality.confidence,
        dominant: languageModality.available === false ? null : dominantEmotion(languageModality.scores),
        detail: (conversation?.timings.transcription_seconds ?? 0) > 0
          ? 'Emotion evidence inferred by Gemma from the ElevenLabs Scribe transcript.'
          : 'Emotion evidence inferred by Gemma from the typed words.',
      }
    : null;
  const demoActive = demoFaceActive || demoProsodyActive;
  const phaseCopy = proofStage === 'baseline'
    ? 'Counterfactual 1/2 — Gemma is answering from the transcript alone…'
    : proofStage === 'adapted'
      ? `Counterfactual 2/2 — adding the injected ${demoEmotion} face + voice context…`
      : phase === 'ready' && !recorderAvailable
        ? judgeProofActive
          ? 'Judge proof ready. Same transcript; only the labeled emotion signal will change.'
          : demoActive
            ? 'Ready with a permanently labeled rehearsal fallback. Typed conversation remains available.'
          : 'Ready. Use the typed fallback below to begin.'
        : PHASE_COPY[phase];
  const showStageStatus = Boolean(
    proofStage
    || phase === 'requesting'
    || phase === 'listening'
    || phase === 'transcribing'
    || phase === 'thinking'
    || phase === 'awaiting-audio'
    || phase === 'speaking'
    || phase === 'holding'
    || phase === 'error'
  );

  return (
    <main className={`app-shell${judgeProofActive ? ' is-proof-mode' : ''}${conversation ? ' has-conversation' : ''}`}>
      <section className="mirror-stage">
        <a className="stage-brand" href="/" aria-label="Null Mirror home">
          <span aria-hidden="true">◌</span>
          <strong>NULL MIRROR</strong>
        </a>
        <div className="stage-badges">
          <a className="explainer-link" href="/how-it-works">HOW IT WORKS</a>
          {demoRequested ? (
            <span className={`demo-badge${demoActive || judgeProofActive ? ' is-active' : ''}`}>
              {proofStage === 'baseline'
                ? 'CONTEXT WITHHELD · CONTROL'
                : judgeProofActive && demoActive
                  ? `INJECTED ${EMOTION_LABELS[demoEmotion]} SIGNAL`
                  : demoActive
                    ? 'LABELED DEMO FALLBACK'
                    : judgeProofActive
                      ? 'JUDGE PROOF READY'
                      : 'JUDGE PROOF OPT-IN'}
            </span>
          ) : null}
        </div>
        {showStageStatus ? (
          <p className={`stage-status stage-status--${phase}`} role="status" aria-live="polite">
            <span className={`phase-light phase-light--${phase}`} aria-hidden="true" />
            {phaseCopy}
          </p>
        ) : null}
        <CrowdPane
          side="user"
          label="01 · human signal"
          title="What the system observed"
          emotion={userCrowd.dominant}
          confidence={userCrowd.confidence}
        >
          {judgeProofActive ? (
            <div className={`proof-signal-inline proof-signal-inline--${demoEmotion}`}>
              <strong>{proofStage === 'baseline' ? 'WORDS ONLY' : EMOTION_LABELS[demoEmotion]}</strong>
              <span>
                {proofStage === 'baseline'
                  ? 'face + voice withheld'
                  : `face ${confidenceLabel(faceSignal.confidence)} · voice ${confidenceLabel(prosodySignal.confidence)} · labeled demo`}
              </span>
            </div>
          ) : (
            <>
              <div className="camera-card">
                <video ref={videoRef} muted playsInline aria-label="Mirrored live camera preview" />
                {!mediaRef.current ? (
                  <div className="camera-empty">
                    <span aria-hidden="true">◎</span>
                    <p>{permissionNote ?? 'Camera preview begins only with your permission.'}</p>
                  </div>
                ) : null}
                <span className={`camera-state camera-state--${faceSignal.status}`}>
                  {faceSignal.status === 'live' ? 'FACE LIVE' : faceSignal.status === 'demo' ? 'DEMO FACE' : faceSignal.status.replace('-', ' ')}
                </span>
              </div>
              <div className="signal-stack">
                <SignalChip label="Face model" signal={faceSignal} />
                <SignalChip label="Voice energy" signal={prosodySignal} caveat="prosody heuristic" />
                {languageSignal ? <SignalChip label="Words · Gemma" signal={languageSignal} caveat="transcript language" /> : null}
              </div>
            </>
          )}
          {conversation?.transcript ? (
            <aside className="human-utterance">
              <span>YOU SAID</span>
              <p>{conversation.transcript}</p>
            </aside>
          ) : null}
          <EmotionHistogram scores={userCrowd.scores} active={userCrowd.active} />
        </CrowdPane>

        <div className="mirror-axis" aria-hidden="true">
          <span style={{ height: `${Math.round(contextAxisValue * 100)}%` }} />
        </div>

        {!judgeProofActive && recorderAvailable ? (
          <div className="conversation-control">
            <span className="context-impact-readout">{contextAxisCopy}</span>
            <button
              type="button"
              className={`talk-button${recording ? ' is-recording' : ''}`}
              onClick={recording ? stopRecording : startRecording}
              disabled={!recorderAvailable || busy || phase === 'awaiting-audio' || phase === 'speaking' || phase === 'holding'}
              aria-pressed={recording}
            >
              <span className="talk-icon" aria-hidden="true">{recording ? '■' : '●'}</span>
              <strong>{recording ? 'Stop & respond' : 'Speak to the mirror'}</strong>
              <small>
                {recording
                  ? `${recordingSeconds}s / ${MAX_RECORDING_SECONDS}s · auto-sends at limit`
                  : recorderAvailable ? 'tap to record · 45s maximum' : 'microphone unavailable'}
              </small>
            </button>
          </div>
        ) : null}

        <CrowdPane
          side="model"
          label="02 · model response"
          title="What Gemma is expressing back"
          emotion={displayedModelEmotion}
          confidence={displayedModelStrength}
          metricLabel="layer-28 alignment"
        >
          {counterfactualProof || responseContext ? (
            <div className={`response-card${conversation ? ' has-response' : ''}`}>
            {counterfactualProof ? (
              <section
                className={`counterfactual-proof counterfactual-proof--${counterfactualProof.emotion}`}
                aria-label="Counterfactual proof"
              >
                <header>
                  <span>CAUSAL PROOF</span>
                  <strong>SAME WORDS · DIFFERENT HUMAN CONTEXT</strong>
                </header>
                <div className="proof-plan-flow">
                  <article className="proof-plan proof-plan--baseline">
                    <span>WORDS ONLY</span>
                    <strong>{proofBaselinePlan}</strong>
                  </article>
                  <i aria-hidden="true">→</i>
                  <article className="proof-plan proof-plan--adapted">
                    <span>+ {EMOTION_LABELS[counterfactualProof.emotion].toUpperCase()} FACE + VOICE</span>
                    <strong>{proofAdaptedPlan}</strong>
                  </article>
                </div>
                <footer>{proofOutcomeCopy}{proofShiftCopy}</footer>
                {displayedModelPhrase ? (
                  <div className="proof-vector-trace">
                    <span>LIVE LAYER-28 TRACE</span>
                    <strong>
                      {displayedModelPhrase.emotion?.toUpperCase()} · {confidenceLabel(displayedModelPhrase.intensity)} ALIGNMENT
                    </strong>
                    <small>{displayedModelPhrase.direction || 'Measured response delivery'}</small>
                  </div>
                ) : null}
                <p className="proof-announcement" role="status" aria-live="polite">
                  {counterfactualProof.adapted
                    ? `Counterfactual result: ${proofBaselinePlan} became ${proofAdaptedPlan}. ${proofOutcomeCopy.toLowerCase()}.`
                    : proofStage === 'failed'
                      ? 'Adapted comparison failed. The transcript-only control is preserved and can be retried.'
                      : 'Transcript-only control captured. Adding the selected nonverbal context.'}
                </p>
              </section>
            ) : responseContext ? (
              <div className="response-context">
                <span>WHY IT MATTERED</span>
                <strong>{strategyLabel}</strong>
                <p>{adaptationCopy}</p>
                <small>{contextDetailCopy}</small>
                <i aria-hidden="true"><b style={{ width: `${Math.round(responseContext.nonverbalWeight * 100)}%` }} /></i>
              </div>
            ) : null}
            </div>
          ) : null}
          {conversation ? (
            <ModelVoice
              conversation={conversation}
              playbackTime={playbackTime}
              playbackDuration={playbackDuration}
              phase={phase}
              dominant={displayedModelEmotion}
              strength={displayedModelStrength}
              needsPlay={needsPlay}
              onPlay={playResponse}
            />
          ) : null}
          <EmotionHistogram
            scores={displayedModelScores}
            active={Boolean(displayedModelEmotion)}
          />
          <audio
            ref={audioRef}
            hidden
            preload="none"
            aria-label="Gemma expressive response"
          />
        </CrowdPane>
      </section>

      <section className="floating-dock" aria-label="Conversation controls">
        <form
          className={`typed-fallback${judgeProofActive ? ' is-proof' : ''}`}
          onSubmit={(event) => {
            event.preventDefault();
            if (judgeProofActive) void runCounterfactualProof();
            else submitTyped();
          }}
        >
          <label className="sr-only" htmlFor="typed-message">
            {judgeProofActive ? 'Pick the missing human context' : 'Typed fallback'}
          </label>
          <div className="conditioning-control">
            <span>AFFECT CHANNEL</span>
            <button
              type="button"
              className={`conditioning-switch${conditioningMode === 'vector' ? ' is-vector' : ''}`}
              role="switch"
              aria-checked={conditioningMode === 'vector'}
              aria-label="Use layer-28 emotion-vector conditioning instead of emotion text in the prompt"
              title={conditioningMode === 'vector'
                ? 'The fused emotion direction is injected into the final user sentence at layer 28.'
                : 'Emotion context is written into Gemma’s prompt.'}
              disabled={textLocked}
              onClick={() => setConditioningMode((current) => current === 'prompt' ? 'vector' : 'prompt')}
            >
              <span>PROMPT</span>
              <i aria-hidden="true"><b /></i>
              <span>VECTOR</span>
            </button>
          </div>
          {judgeProofActive ? (
            <div className="demo-emotion-rail" aria-label="Injected emotion preset">
              {SHARED_EMOTIONS.map((emotion) => (
                <button
                  key={emotion}
                  type="button"
                  className={`demo-emotion demo-emotion--${emotion}${demoEmotion === emotion ? ' is-selected' : ''}`}
                  aria-pressed={demoEmotion === emotion}
                  disabled={textLocked}
                  onClick={() => chooseDemoEmotion(emotion)}
                >
                  {EMOTION_LABELS[emotion]}
                </button>
              ))}
            </div>
          ) : null}
          <div>
            <input
              id="typed-message"
              value={typedText}
              onChange={(event) => setTypedText(event.target.value)}
              placeholder={judgeProofActive ? DEMO_PROOF_LINE : 'Type what you want Gemma to respond to…'}
              disabled={textLocked}
              readOnly={judgeProofActive}
            />
            <button type="submit" disabled={!typedText.trim() || textLocked}>
              {judgeProofActive ? 'Prove it' : 'Send'}
            </button>
          </div>
        </form>
      </section>

      {error || voiceNotice ? (
        <aside className="notice" role="alert">
          <strong>{error ? 'Pipeline unavailable' : 'Voice note'}</strong>
          <span>{error ?? voiceNotice}</span>
          <button type="button" onClick={() => { setError(null); setVoiceNotice(null); if (phase === 'error') setPhase('ready'); }}>Dismiss</button>
        </aside>
      ) : null}

      {!started ? (
        <section className="consent-gate" role="dialog" aria-modal="true" aria-labelledby="consent-title">
          <div className="consent-glow" aria-hidden="true" />
          <div className="consent-card">
            {demoRequested ? <span className="demo-badge is-active">DEMO SIGNAL OPT-IN</span> : null}
            <p className="consent-kicker">
              {demoRequested ? 'THE CONTEXT SPEECH-TO-TEXT DELETES' : 'A TWO-SIDED EMOTION CONVERSATION'}
            </p>
            <h1 id="consent-title">
              {demoRequested ? (
                <>Same words. Different human.<br />Different answer.</>
              ) : (
                <>Let the model hear<br />more than words.</>
              )}
            </h1>
            <p className="consent-copy">
              {demoRequested ? (
                <>
                  Run one transcript twice: first as words alone, then with face and voice context.
                  See exactly when that missing human signal changes Gemma’s response plan, reply,
                  internal emotion vectors, simulated crowd, and ElevenLabs voice.
                </>
              ) : (
                <>
                  Live facial expression, browser-estimated voice energy, and transcript become one
                  transparent context. Gemma uses it to choose a response plan; the measured internal
                  vector trace then drives the simulated crowd and ElevenLabs delivery.
                </>
              )}
            </p>
            <ul>
              <li><span>Camera</span><span className="consent-detail">Downscaled JPEG frames go to the local face endpoint about 4–5 times per second.</span></li>
              <li><span>Microphone</span><span className="consent-detail">Live energy stays in this browser. On stop, recorded audio goes through the local backend to ElevenLabs Scribe.</span></li>
              <li><span>Honesty</span><span className="consent-detail">Missing services show as unavailable. Demo data runs only with <code>?demo=1</code>.</span></li>
            </ul>
            <div className="consent-actions">
              {demoRequested ? (
                <>
                  <button type="button" className="consent-primary" onClick={openJudgeProof} disabled={phase === 'requesting'}>
                    Open judge proof
                  </button>
                  <button type="button" className="consent-secondary" onClick={() => void startExperience()} disabled={phase === 'requesting'}>
                    {phase === 'requesting' ? 'Requesting permission…' : 'Use live camera + microphone'}
                  </button>
                </>
              ) : (
                <>
                  <button type="button" className="consent-primary" onClick={() => void startExperience()} disabled={phase === 'requesting'}>
                    {phase === 'requesting' ? 'Requesting permission…' : 'Start camera + microphone'}
                  </button>
                  <button type="button" className="consent-secondary" onClick={continueTyped} disabled={phase === 'requesting'}>Continue with typing</button>
                </>
              )}
            </div>
            <small>
              {demoRequested
                ? 'Judge proof uses clearly labeled synthetic signals; the live path remains one click away.'
                : 'Nothing starts automatically. You stay in control of recording.'}
            </small>
          </div>
        </section>
      ) : null}
    </main>
  );
}
