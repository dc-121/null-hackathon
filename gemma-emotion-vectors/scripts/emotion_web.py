#!/usr/bin/env python3
"""Local technical UI for Gemma activation-to-voice transfer."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import threading
import uuid
import webbrowser

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import uvicorn

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
    resolve_api_key,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_DIR / "web"


class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)
    max_new_tokens: int = Field(default=64, ge=1, le=160)
    delivery_mode: str = Field(default="raw", pattern="^(raw|safe)$")
    phrase_tokens: int = Field(default=6, ge=3, le=16)


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
        self.speech: SpeechClient | None = None
        self.jobs: OrderedDict[str, str] = OrderedDict()


runtime = Runtime()
app = FastAPI(title="Gemma emotion-to-voice lab")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {
        "model": "google/gemma-4-E4B-it",
        "layer": 28,
        "emotions": runtime.names,
        "raw_tags": RAW_INTENSITY_TAGS,
        "raw_extreme_tags": RAW_EXTREME_TAGS,
        "safe_tags": SAFE_SPEECH_TAGS,
        "selection": {
            "minimum_evidence": MIN_VOICE_EVIDENCE,
            "switch_ratio": EMOTION_SWITCH_RATIO,
            "switch_margin": EMOTION_SWITCH_MARGIN,
        },
    }


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

    speech_id = uuid.uuid4().hex
    runtime.jobs[speech_id] = tagged_text
    while len(runtime.jobs) > 20:
        runtime.jobs.popitem(last=False)

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


@app.get("/api/audio/{speech_id}")
def audio(speech_id: str) -> StreamingResponse:
    tagged_text = runtime.jobs.get(speech_id)
    if tagged_text is None:
        raise HTTPException(status_code=404, detail="Speech plan expired")
    if runtime.speech is None:
        raise HTTPException(status_code=503, detail="ElevenLabs is unavailable")

    def stream():
        with runtime.audio_lock:
            yield from runtime.speech.iter_audio(tagged_text)

    return StreamingResponse(stream(), media_type="audio/mpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_DIR / ".env")
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
    runtime.speech = SpeechClient(
        api_key=resolve_api_key(),
        voice_id=DEFAULT_VOICE_ID,
        model_id=DEFAULT_TTS_MODEL,
        output_format=DEFAULT_OUTPUT_FORMAT,
        stability=0.25,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Model ready in {load_seconds:.2f}s · opening {url}")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if runtime.speech is not None:
            runtime.speech.close()


if __name__ == "__main__":
    main()
