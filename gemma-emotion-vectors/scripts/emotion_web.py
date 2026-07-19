#!/usr/bin/env python3
"""Local technical UI for Gemma activation-to-voice transfer."""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import OrderedDict
from dataclasses import asdict
import logging
import os
from pathlib import Path
import socket
import threading
from time import perf_counter
from typing import Any, Literal
import uuid
import webbrowser

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch
import uvicorn

try:
    from emotion_api import (
        MAX_AUDIO_BYTES,
        LANGUAGE_CONFIDENCE_CAP,
        MIN_PROMPT_MODALITY_CONFIDENCE,
        FaceEmotionAnalyzer,
        FaceInferenceUnavailable,
        Modality,
        SHARED_EMOTIONS,
        add_phrase_timings,
        explain_response_context,
        fuse_modalities,
        response_prompt,
        score_language_emotion,
        shared_phrase_plan,
        validate_input_scores,
    )
    from emotion_trace import (
        DEFAULT_OUTPUT_FORMAT,
        DEFAULT_TTS_MODEL,
        DEFAULT_VOICE_ID,
        EMOTION_SWITCH_MARGIN,
        EMOTION_SWITCH_RATIO,
        MIN_VOICE_EVIDENCE,
        RAW_EXTREME_TAGS,
        RAW_INTENSITY_TAGS,
        SAFE_SPEECH_TAGS,
        SpeechClient,
        analyze_prompt,
        build_tagged_speech,
        load_runtime,
    )
except ModuleNotFoundError:  # imported as scripts.emotion_web in tests
    from scripts.emotion_api import (
        MAX_AUDIO_BYTES,
        LANGUAGE_CONFIDENCE_CAP,
        MIN_PROMPT_MODALITY_CONFIDENCE,
        FaceEmotionAnalyzer,
        FaceInferenceUnavailable,
        Modality,
        SHARED_EMOTIONS,
        add_phrase_timings,
        explain_response_context,
        fuse_modalities,
        response_prompt,
        score_language_emotion,
        shared_phrase_plan,
        validate_input_scores,
    )
    from scripts.emotion_trace import (
        DEFAULT_OUTPUT_FORMAT,
        DEFAULT_TTS_MODEL,
        DEFAULT_VOICE_ID,
        EMOTION_SWITCH_MARGIN,
        EMOTION_SWITCH_RATIO,
        MIN_VOICE_EVIDENCE,
        RAW_EXTREME_TAGS,
        RAW_INTENSITY_TAGS,
        SAFE_SPEECH_TAGS,
        SpeechClient,
        analyze_prompt,
        build_tagged_speech,
        load_runtime,
    )


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIENCE_DIST_DIR = PROJECT_DIR.parent / "experience" / "dist"
LEGACY_WEB_DIR = PROJECT_DIR / "web"
WEB_DIR = (
    EXPERIENCE_DIST_DIR
    if (EXPERIENCE_DIST_DIR / "index.html").is_file()
    else LEGACY_WEB_DIR
)
LOGGER = logging.getLogger("emotion-mirror")
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "video/webm",
}
SharedEmotion = Literal["happy", "sad", "angry", "afraid", "surprised"]
API_VERSION = "2026-07-18.1"


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4_000)
    max_new_tokens: int = Field(default=64, ge=1, le=160)
    delivery_mode: str = Field(default="raw", pattern="^(raw|safe)$")
    phrase_tokens: int = Field(default=6, ge=3, le=16)


class ConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["2026-07-18.1"]
    transcript: str | None = Field(default=None, max_length=4_000)
    audio_base64: str | None = Field(default=None, max_length=36_000_000)
    audio_content_type: str | None = Field(default=None, max_length=80)
    face_scores: dict[str, float] | None = None
    face_confidence: float | None = Field(default=None, ge=0, le=1)
    prosody_scores: dict[str, float] | None = None
    prosody_confidence: float | None = Field(default=None, ge=0, le=1)
    # Counterfactual proof runs still need the real response/vector trace, but
    # only the adapted side should spend latency and quota on ElevenLabs audio.
    synthesize_speech: bool = True

    @field_validator("face_scores", "prosody_scores")
    @classmethod
    def valid_scores(
        cls, values: dict[str, float] | None
    ) -> dict[str, float] | None:
        if values is None:
            return None
        # Validate here and preserve the caller's values. Fusion normalizes once.
        validate_input_scores(values)
        return values

    @model_validator(mode="after")
    def valid_sources(self) -> "ConversationRequest":
        has_transcript = bool(self.transcript and self.transcript.strip())
        has_audio = bool(self.audio_base64 and self.audio_base64.strip())
        if has_transcript == has_audio:
            raise ValueError("provide exactly one of transcript or audio_base64")
        if has_audio:
            content_type = (self.audio_content_type or "").split(";", 1)[0].lower()
            if content_type not in ALLOWED_AUDIO_TYPES:
                raise ValueError("audio_content_type is required and unsupported")
            self.audio_content_type = content_type
        elif self.audio_content_type is not None:
            raise ValueError("audio_content_type requires audio_base64")

        for name in ("face", "prosody"):
            scores = getattr(self, f"{name}_scores")
            confidence = getattr(self, f"{name}_confidence")
            if (scores is None) != (confidence is None):
                raise ValueError(
                    f"{name}_scores and {name}_confidence must be provided together"
                )
        return self


class EmotionScoresResponse(BaseModel):
    happy: float = Field(ge=0)
    sad: float = Field(ge=0)
    angry: float = Field(ge=0)
    afraid: float = Field(ge=0)
    surprised: float = Field(ge=0)


class FaceResponse(BaseModel):
    detected: bool
    scores: EmotionScoresResponse
    retained_mass: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    dominant: SharedEmotion | None


class ModalityResponse(BaseModel):
    scores: EmotionScoresResponse
    confidence: float = Field(ge=0, le=1)
    retained_mass: float = Field(ge=0, le=1)


class HumanResponse(BaseModel):
    scores: EmotionScoresResponse
    dominant: SharedEmotion | None
    confidence: float = Field(ge=0, le=1)
    modalities: dict[str, ModalityResponse]


class PhraseResponse(BaseModel):
    text: str
    emotion: SharedEmotion | None
    scores: EmotionScoresResponse
    intensity: float = Field(ge=0, le=1)
    evidence: float = Field(ge=0)
    shared_mass: float = Field(ge=0, le=1)
    direction: str
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)


class ConversationTimings(BaseModel):
    transcription_seconds: float = Field(ge=0)
    language_seconds: float = Field(ge=0)
    generation_seconds: float = Field(ge=0)
    replay_seconds: float = Field(ge=0)
    speech_seconds: float | None = Field(default=None, ge=0)
    total_seconds: float = Field(ge=0)


class ResponseContextResponse(BaseModel):
    dominant: SharedEmotion | None
    confidence: float = Field(ge=0, le=1)
    nonverbal_weight: float = Field(ge=0, le=1)
    nonverbal_shift: float | None = Field(default=None, ge=0, le=1)
    nonverbal_dominant: SharedEmotion | None
    language_dominant: SharedEmotion | None
    effect: Literal[
        "words-only",
        "reinforced",
        "adjusted",
        "shifted",
        "nonverbal-only",
        "language-mixed",
        "mixed",
    ]
    sources: list[Literal["face", "prosody"]]
    source_dominants: dict[str, SharedEmotion | None]
    strategy: Literal[
        "celebrate",
        "support",
        "de-escalate",
        "reassure",
        "orient",
        "stay-curious",
    ]


class ConversationResponse(BaseModel):
    api_version: Literal["2026-07-18.1"]
    transcript: str
    human: HumanResponse
    response_context: ResponseContextResponse
    response: str
    speech_id: str | None
    phrases: list[PhraseResponse]
    timings: ConversationTimings


class Runtime:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.names: list[str] = []
        self.vectors: torch.Tensor | None = None
        self.neutral_mean: torch.Tensor | None = None
        self.neutral_std: torch.Tensor | None = None
        self.model_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.speech: SpeechClient | None = None
        self.jobs: OrderedDict[str, str | bytes] = OrderedDict()
        self.diagnostics: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.face = FaceEmotionAnalyzer()


runtime = Runtime()
app = FastAPI(title="Gemma emotion-to-voice lab")
if WEB_DIR == EXPERIENCE_DIST_DIR and (WEB_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
else:
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/how-it-works")
def how_it_works() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {
        "api_version": API_VERSION,
        "model": "google/gemma-4-E4B-it",
        "layer": 28,
        "emotions": runtime.names,
        "shared_emotions": SHARED_EMOTIONS,
        "raw_tags": RAW_INTENSITY_TAGS,
        "raw_extreme_tags": RAW_EXTREME_TAGS,
        "safe_tags": SAFE_SPEECH_TAGS,
        "selection": {
            "minimum_evidence": MIN_VOICE_EVIDENCE,
            "switch_ratio": EMOTION_SWITCH_RATIO,
            "switch_margin": EMOTION_SWITCH_MARGIN,
            "language_confidence_cap": LANGUAGE_CONFIDENCE_CAP,
            "minimum_prompt_modality_confidence": MIN_PROMPT_MODALITY_CONFIDENCE,
        },
    }


def store_job(value: str | bytes) -> str:
    job_id = uuid.uuid4().hex
    with runtime.jobs_lock:
        runtime.jobs[job_id] = value
        while len(runtime.jobs) > 20:
            runtime.jobs.popitem(last=False)
    return job_id


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    if (
        runtime.model is None
        or runtime.tokenizer is None
        or runtime.vectors is None
        or runtime.neutral_mean is None
        or runtime.neutral_std is None
    ):
        raise HTTPException(status_code=503, detail="Gemma is still loading")

    with runtime.model_lock:
        result = analyze_prompt(
            request.prompt,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            names=runtime.names,
            vectors=runtime.vectors,
            neutral_mean=runtime.neutral_mean,
            neutral_std=runtime.neutral_std,
            max_new_tokens=request.max_new_tokens,
        )
        tagged_text, decisions = build_tagged_speech(
            result,
            runtime.tokenizer,
            delivery_mode=request.delivery_mode,
            phrase_tokens=request.phrase_tokens,
        )

    speech_id = store_job(tagged_text)

    return {
        "response": result.response,
        "tagged_text": tagged_text,
        "speech_id": speech_id,
        "baseline": {
            name: float(result.scores[0, index])
            for index, name in enumerate(result.names)
        },
        "phrases": [asdict(decision) for decision in decisions],
        "timings": {
            "generation_seconds": result.generation_seconds,
            "replay_seconds": result.replay_seconds,
        },
    }


@app.post("/api/face", response_model=FaceResponse)
def face(jpeg: bytes = Body(media_type="image/jpeg")) -> dict:
    """Classify the largest face in a raw JPEG without retaining the image."""
    try:
        return runtime.face.analyze(jpeg)
    except FaceInferenceUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        LOGGER.exception("Face inference failed")
        raise HTTPException(status_code=502, detail="Face inference failed") from error


def decode_audio(request: ConversationRequest) -> bytes:
    assert request.audio_base64 is not None
    try:
        audio = base64.b64decode(request.audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail="audio_base64 is invalid") from error
    if not audio:
        raise HTTPException(status_code=422, detail="audio_base64 is empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio exceeds 25 MiB")
    return audio


@app.post("/api/conversation", response_model=ConversationResponse)
def conversation(request: ConversationRequest) -> dict:
    started = perf_counter()
    if (
        runtime.model is None
        or runtime.tokenizer is None
        or runtime.vectors is None
        or runtime.neutral_mean is None
        or runtime.neutral_std is None
    ):
        raise HTTPException(status_code=503, detail="Gemma is still loading")

    transcription_seconds = 0.0
    if request.audio_base64 is not None:
        if runtime.speech is None:
            raise HTTPException(
                status_code=503,
                detail="ElevenLabs is required to transcribe audio",
            )
        audio = decode_audio(request)
        transcription_started = perf_counter()
        try:
            with runtime.audio_lock:
                transcription = runtime.speech.transcribe(
                    audio, content_type=request.audio_content_type or "audio/webm"
                )
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            LOGGER.exception("Scribe v2 transcription failed")
            raise HTTPException(
                status_code=502, detail="Scribe v2 transcription failed"
            ) from error
        transcript = str(transcription["text"]).strip()
        transcription_seconds = perf_counter() - transcription_started
    else:
        transcript = (request.transcript or "").strip()

    language_started = perf_counter()
    with runtime.model_lock:
        language_scores, language_confidence, language_diagnostics = (
            score_language_emotion(
                transcript,
                model=runtime.model,
                tokenizer=runtime.tokenizer,
                names=runtime.names,
                vectors=runtime.vectors,
                neutral_mean=runtime.neutral_mean,
                neutral_std=runtime.neutral_std,
            )
        )
        # The current transcript-vector scorer is useful supporting evidence,
        # but the five-prompt calibration suite shows it is not reliable enough
        # to overpower a good face observation. Keep it visibly contributory
        # while letting the user's nonverbal signal change Gemma's response.
        language_confidence = min(
            language_confidence,
            LANGUAGE_CONFIDENCE_CAP,
        )
        language_seconds = perf_counter() - language_started
        modalities = [
            Modality(
                "language",
                language_scores,
                language_confidence,
                retained_mass=sum(language_scores.values()),
            )
        ]
        if (
            request.face_scores is not None
            and (request.face_confidence or 0.0)
            >= MIN_PROMPT_MODALITY_CONFIDENCE
        ):
            modalities.append(
                Modality(
                    "face",
                    request.face_scores,
                    request.face_confidence or 0.0,
                )
            )
        if (
            request.prosody_scores is not None
            and (request.prosody_confidence or 0.0)
            >= MIN_PROMPT_MODALITY_CONFIDENCE
        ):
            modalities.append(
                Modality(
                    "prosody",
                    request.prosody_scores,
                    request.prosody_confidence or 0.0,
                )
            )
        fused = fuse_modalities(modalities)
        response_context = explain_response_context(fused)
        result = analyze_prompt(
            response_prompt(transcript, fused),
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            names=runtime.names,
            vectors=runtime.vectors,
            neutral_mean=runtime.neutral_mean,
            neutral_std=runtime.neutral_std,
            max_new_tokens=96,
        )
        tagged_text, phrases, text_spans = shared_phrase_plan(
            result,
            runtime.tokenizer,
            strategy=response_context["strategy"],
        )

    speech_id: str | None = None
    speech_seconds: float | None = None
    if request.synthesize_speech and runtime.speech is not None and tagged_text:
        speech_started = perf_counter()
        try:
            with runtime.audio_lock:
                speech_audio, alignment = (
                    runtime.speech.synthesize_with_timestamps(tagged_text)
                )
            speech_seconds = perf_counter() - speech_started
            add_phrase_timings(phrases, text_spans, alignment)
            speech_id = store_job(speech_audio)
        except (httpx.HTTPError, RuntimeError, ValueError):
            # A response without audio remains useful and keeps the live demo
            # moving. Null speech_id and phrase times make the degradation
            # explicit to the browser without leaking provider error details.
            LOGGER.exception("ElevenLabs v3 speech generation failed")

    diagnostic_id = speech_id or uuid.uuid4().hex
    response_baseline = {
        name: float(result.scores[0, index])
        for index, name in enumerate(result.names)
    }
    response_trace = [
        {
            name: float(row[index])
            for index, name in enumerate(result.names)
        }
        for row in result.scores
    ]
    with runtime.jobs_lock:
        runtime.diagnostics[diagnostic_id] = {
            "language": language_diagnostics,
            "response_context": response_context,
            "response_baseline": response_baseline,
            "response_trace": response_trace,
        }
        while len(runtime.diagnostics) > 20:
            runtime.diagnostics.popitem(last=False)

    return {
        "api_version": API_VERSION,
        "transcript": transcript,
        "human": {
            "scores": fused.scores,
            "dominant": fused.dominant,
            "confidence": fused.confidence,
            "modalities": fused.modalities,
        },
        "response_context": response_context,
        "response": result.response,
        "speech_id": speech_id,
        "phrases": phrases,
        "timings": {
            "transcription_seconds": transcription_seconds,
            "language_seconds": language_seconds,
            "generation_seconds": result.generation_seconds,
            "replay_seconds": result.replay_seconds,
            "speech_seconds": speech_seconds,
            "total_seconds": perf_counter() - started,
        },
    }


@app.get("/api/audio/{speech_id}")
def audio(speech_id: str) -> StreamingResponse:
    with runtime.jobs_lock:
        job = runtime.jobs.get(speech_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Speech plan expired")
    if isinstance(job, bytes):
        return StreamingResponse(
            iter((job,)),
            media_type="audio/mpeg",
            headers={"Cache-Control": "private, no-store"},
        )
    if runtime.speech is None:
        raise HTTPException(status_code=503, detail="ElevenLabs is unavailable")

    def stream():
        with runtime.audio_lock:
            yield from runtime.speech.iter_audio(job)

    return StreamingResponse(stream(), media_type="audio/mpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def reserve_server_socket(host: str, port: int) -> socket.socket:
    """Bind the HTTP port before loading Gemma and keep it reserved."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    server_socket = socket.socket(family, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((host, port))
    except OSError as error:
        server_socket.close()
        raise SystemExit(
            f"Cannot start server on {host}:{port}: {error}"
        ) from error
    server_socket.set_inheritable(True)
    return server_socket


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_DIR / ".env")
    server_socket = reserve_server_socket(args.host, args.port)
    try:
        (
            model,
            tokenizer,
            names,
            vectors,
            neutral_mean,
            neutral_std,
            load_seconds,
        ) = load_runtime()
        runtime.model = model
        runtime.tokenizer = tokenizer
        runtime.names = names
        runtime.vectors = vectors
        runtime.neutral_mean = neutral_mean
        runtime.neutral_std = neutral_std
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if api_key:
            runtime.speech = SpeechClient(
                api_key=api_key,
                voice_id=DEFAULT_VOICE_ID,
                model_id=DEFAULT_TTS_MODEL,
                output_format=DEFAULT_OUTPUT_FORMAT,
                stability=0.25,
            )
        else:
            LOGGER.warning(
                "ELEVENLABS_API_KEY is unset; typed conversation works without "
                "transcription or speech"
            )
        url = f"http://{args.host}:{args.port}"
        frontend = "Null Mirror" if WEB_DIR == EXPERIENCE_DIST_DIR else "legacy lab"
        print(
            f"Model ready in {load_seconds:.2f}s · serving {frontend} · opening {url}"
        )
        if not args.no_open:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level="warning",
            )
        )
        server.run(sockets=[server_socket])
    finally:
        server_socket.close()
        if runtime.speech is not None:
            runtime.speech.close()


if __name__ == "__main__":
    main()
