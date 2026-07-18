#!/usr/bin/env python3
"""Trace Gemma emotion vectors per token and optionally speak them expressively."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator
from dataclasses import dataclass
import getpass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter

import httpx
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-4-E4B-it"
VECTOR_REPO = "rain1955/emotion-vector-replication"
VECTOR_FILE = "results/emotion_vectors.npz"
TARGET_LAYER = 28

ELEVENLABS_API = "https://api.elevenlabs.io"
DEFAULT_TTS_MODEL = "eleven_v3"
DEFAULT_VOICE_ID = "N2lVS1w4EtoT3dr4eOWO"  # Callum: expressive character voice
DEFAULT_OUTPUT_FORMAT = "mp3_22050_32"
ABSOLUTE_Z_FLOOR = 0.5
DELTA_Z_FLOOR = 0.25
DELTA_EVIDENCE_SCALE = 0.35
MAX_INTENSITY_EVIDENCE = 6.0
MIN_VOICE_EVIDENCE = 1.25
EMOTION_SWITCH_RATIO = 1.20
EMOTION_SWITCH_MARGIN = 0.50

NEUTRAL_CALIBRATION_TEXTS = (
    "The package contains four identical metal brackets and eight screws.",
    "A calendar lists the weekdays from Monday through Sunday.",
    "The document has twelve pages and a table on the final page.",
    "Water freezes at zero degrees Celsius under standard pressure.",
    "The train departs from platform three at nine in the morning.",
    "A rectangle has two pairs of parallel sides.",
    "The database table contains an integer identifier and a timestamp.",
    "The road continues north for two kilometers before the intersection.",
    "The device uses a rechargeable battery and a standard USB connector.",
    "The report groups the measurements by date and geographic region.",
    "The room contains a desk, two chairs, and a closed storage cabinet.",
    "The function accepts a string and returns the number of characters.",
)

SAFE_SPEECH_TAGS = {
    "afraid": ("concerned",),
    "angry": ("frustrated",),
    "calm": ("calm",),
    "desperate": ("urgent",),
    "guilty": ("remorseful",),
    "happy": ("excited",),
    "loving": ("warmly",),
    "sad": ("sorrowful",),
    "surprised": ("surprised",),
}

RAW_INTENSITY_TAGS = {
    "afraid": ("uneasy", "fearful", "terrified"),
    "angry": ("irritated", "angry", "furious"),
    "calm": ("gentle", "calm", "deeply calm"),
    "desperate": ("urgent", "desperate", "desperate"),
    "guilty": ("uneasy", "guilty", "remorseful"),
    "happy": ("pleased", "excited", "ecstatic"),
    "loving": ("warmly", "affectionately", "lovingly"),
    "sad": ("somber", "sad", "crying"),
    "surprised": ("curious", "surprised", "shocked"),
}

RAW_EXTREME_TAGS = {
    "afraid": "voice trembling",
    "angry": "shouts",
    "calm": "slow",
    "desperate": "pleading",
    "guilty": "voice breaking",
    "happy": "laughing",
    "loving": "tenderly",
    "sad": "sobbing",
    "surprised": "gasps",
}


@dataclass(frozen=True)
class TraceResult:
    response: str
    response_ids: list[int]
    labels: list[str]
    names: list[str]
    scores: torch.Tensor
    neutral_mean: torch.Tensor
    neutral_std: torch.Tensor
    generation_seconds: float
    replay_seconds: float


@dataclass(frozen=True)
class EmotionComponent:
    name: str
    weight: float
    evidence: float
    raw: float
    delta: float
    z: float
    delta_z: float
    intensity: float


@dataclass(frozen=True)
class SpeechDecision:
    text: str
    start_token: int
    end_token: int
    signals: tuple[EmotionComponent, ...]
    components: tuple[EmotionComponent, ...]
    direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace Gemma's layer-28 activation against published emotion vectors "
            "and optionally speak the response with ElevenLabs."
        )
    )
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Load Gemma once and analyze prompts until Ctrl-D or /quit.",
    )
    parser.add_argument(
        "--timings", action="store_true", help="Print model load and inference timings."
    )
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument(
        "--delivery-mode",
        choices=("raw", "safe"),
        default="raw",
        help="Use theatrical raw emotions or restrained assistant-safe delivery.",
    )
    parser.add_argument(
        "--tts-stability",
        type=float,
        default=0.25,
        help="Eleven v3 stability; lower is more expressive and less predictable.",
    )
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument(
        "--phrase-tokens",
        type=int,
        default=6,
        help="Maximum tokens between voice-control updates.",
    )
    parser.add_argument("--save-audio", type=Path)
    return parser.parse_args()


def visible_token(tokenizer, token_id: int) -> str:
    value = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return value.replace("\n", "\\n").replace("\t", "\\t") or "<empty>"


def load_runtime():
    started = perf_counter()
    vector_path = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
    archive = np.load(vector_path)
    names = sorted(archive.files)
    vectors = torch.from_numpy(np.stack([archive[name] for name in names])).float()
    vectors = torch.nn.functional.normalize(vectors, dim=-1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32
    print(f"Loading {MODEL_ID} on {device} ({dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    neutral_mean, neutral_std = calibrate_neutral(model, tokenizer, vectors)
    return (
        model,
        tokenizer,
        names,
        vectors,
        neutral_mean,
        neutral_std,
        perf_counter() - started,
    )


def calibrate_neutral(model, tokenizer, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure each vector's token-level neutral distribution in this runtime."""
    device = next(model.parameters()).device
    user_message = {
        "role": "user",
        "content": "Provide one plain factual sentence without emotional language.",
    }
    prefix = tokenizer.apply_chat_template(
        [user_message], add_generation_prompt=True, return_dict=True
    )["input_ids"]
    examples: list[dict[str, list[int]]] = []
    selected_positions: list[list[int]] = []
    special_ids = set(tokenizer.all_special_ids)
    for text in NEUTRAL_CALIBRATION_TEXTS:
        input_ids = tokenizer.apply_chat_template(
            [user_message, {"role": "assistant", "content": text}],
            add_generation_prompt=False,
            return_dict=True,
        )["input_ids"]
        examples.append(
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        )
        selected_positions.append(
            [
                index
                for index in range(len(prefix), len(input_ids))
                if input_ids[index] not in special_ids
            ]
        )
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[row, :length] = torch.tensor(
            example["input_ids"], dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
    encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().float().cpu()

    language_model = getattr(model.model, "language_model", model.model)
    handle = language_model.layers[TARGET_LAYER].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(**encoded)
    finally:
        handle.remove()

    mask = torch.zeros_like(encoded["attention_mask"], device="cpu", dtype=torch.bool)
    for row, positions in enumerate(selected_positions):
        mask[row, positions] = True
    hidden = torch.nn.functional.normalize(captured["hidden"][mask], dim=-1)
    neutral_scores = hidden @ vectors.T
    return neutral_scores.mean(dim=0), neutral_scores.std(dim=0, unbiased=False).clamp_min(0.005)


def analyze_prompt(
    prompt: str,
    *,
    model,
    tokenizer,
    names: list[str],
    vectors: torch.Tensor,
    neutral_mean: torch.Tensor,
    neutral_std: torch.Tensor,
    max_new_tokens: int,
) -> TraceResult:
    generation_started = perf_counter()
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt.strip()}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_seconds = perf_counter() - generation_started

    response_tensor = output_ids[0, prompt_length:]
    response_ids = [int(token_id) for token_id in response_tensor]
    response = tokenizer.decode(response_tensor, skip_special_tokens=True).strip()

    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().float().cpu()

    language_model = getattr(model.model, "language_model", model.model)
    handle = language_model.layers[TARGET_LAYER].register_forward_hook(capture)
    replay_started = perf_counter()
    try:
        with torch.inference_mode():
            model(input_ids=output_ids, attention_mask=torch.ones_like(output_ids))
    finally:
        handle.remove()
    replay_seconds = perf_counter() - replay_started

    hidden = captured["hidden"][0]
    positions = [prompt_length - 1] + list(
        range(prompt_length, prompt_length + len(response_ids))
    )
    labels = ["<before response>"] + [
        visible_token(tokenizer, token_id) for token_id in response_ids
    ]
    normalized_hidden = torch.nn.functional.normalize(hidden[positions], dim=-1)
    scores = normalized_hidden @ vectors.T

    return TraceResult(
        response=response,
        response_ids=response_ids,
        labels=labels,
        names=names,
        scores=scores,
        neutral_mean=neutral_mean,
        neutral_std=neutral_std,
        generation_seconds=generation_seconds,
        replay_seconds=replay_seconds,
    )


def print_trace(result: TraceResult, *, top_k: int, timings: bool) -> None:
    print("\n=== Gemma response ===")
    print(result.response)
    print("\n=== Emotion-vector trace (Gemma layer 28) ===")
    print("Cosine activations are directional scores, not probabilities.")
    shown = min(top_k, len(result.names))
    for index, (label, row) in enumerate(
        zip(result.labels, result.scores, strict=True)
    ):
        values, indices = row.topk(shown)
        ranked = ", ".join(
            f"{result.names[int(emotion_index)]}={float(value):+.4f}"
            for value, emotion_index in zip(values, indices, strict=True)
        )
        print(f"[{index:02d}] {label!r}: {ranked}")
    if timings:
        print(
            f"\nTimings: generation {result.generation_seconds:.2f}s · "
            f"activation replay {result.replay_seconds:.2f}s"
        )


def phrase_spans(
    tokenizer, token_ids: list[int], *, max_tokens: int
) -> list[tuple[int, int]]:
    if not token_ids:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for end, token_id in enumerate(token_ids, start=1):
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        length = end - start
        boundary = any(mark in piece for mark in (",", ";", ":", ".", "?", "!", "\n"))
        if (boundary and length >= 3) or length >= max_tokens:
            spans.append((start, end))
            start = end
    if start < len(token_ids):
        spans.append((start, len(token_ids)))
    return spans


def emotion_evidence(
    mean_score: torch.Tensor,
    baseline: torch.Tensor,
    neutral_mean: torch.Tensor,
    neutral_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create comparable control evidence without claiming probabilities."""
    delta = mean_score - baseline
    z = (mean_score - neutral_mean) / neutral_std
    baseline_z = (baseline - neutral_mean) / neutral_std
    delta_z = z - baseline_z
    absolute = (z - ABSOLUTE_Z_FLOOR).clamp_min(0)
    relative = (delta_z - DELTA_Z_FLOOR).clamp_min(0)
    evidence = absolute + relative * DELTA_EVIDENCE_SCALE
    return delta, z, delta_z, evidence


def intensity_level(value: float) -> int:
    if value < 0.34:
        return 0
    if value < 0.67:
        return 1
    return 2


def compile_direction(
    component: EmotionComponent | None, *, delivery_mode: str
) -> str:
    if component is None:
        return ""
    if delivery_mode == "safe":
        tags = list(SAFE_SPEECH_TAGS[component.name])
    else:
        level = intensity_level(component.intensity)
        tags = [RAW_INTENSITY_TAGS[component.name][level]]
        if component.intensity >= 0.85:
            tags.append(RAW_EXTREME_TAGS[component.name])
    return " ".join(f"[{tag}]" for tag in tags)


def select_voice_emotion(
    evidence: torch.Tensor,
    names: list[str],
    previous_name: str | None,
) -> int | None:
    """Select one strong direction and resist weak phrase-to-phrase flicker."""
    winner_index = int(torch.argmax(evidence))
    winner_evidence = float(evidence[winner_index])
    if winner_evidence < MIN_VOICE_EVIDENCE:
        return None
    if previous_name is None or names[winner_index] == previous_name:
        return winner_index

    previous_index = names.index(previous_name)
    previous_evidence = float(evidence[previous_index])
    if previous_evidence < MIN_VOICE_EVIDENCE:
        return winner_index

    switch_floor = max(
        previous_evidence * EMOTION_SWITCH_RATIO,
        previous_evidence + EMOTION_SWITCH_MARGIN,
    )
    return winner_index if winner_evidence >= switch_floor else previous_index


def build_tagged_speech(
    result: TraceResult,
    tokenizer,
    *,
    delivery_mode: str,
    phrase_tokens: int,
) -> tuple[str, list[SpeechDecision]]:
    """Compile continuous activation evidence into one voice direction per phrase."""
    baseline = result.scores[0]
    segments: list[str] = []
    decisions: list[SpeechDecision] = []
    previous_name: str | None = None

    for start, end in phrase_spans(
        tokenizer, result.response_ids, max_tokens=phrase_tokens
    ):
        text = tokenizer.decode(
            result.response_ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not text:
            continue
        mean_score = result.scores[start + 1 : end + 1].mean(dim=0)
        delta, z, delta_z, evidence = emotion_evidence(
            mean_score,
            baseline,
            result.neutral_mean,
            result.neutral_std,
        )
        total = float(evidence.sum())
        weights = evidence / total if total > 0 else torch.zeros_like(evidence)
        signals = tuple(
            EmotionComponent(
                name=name,
                weight=float(weights[index]),
                evidence=float(evidence[index]),
                raw=float(mean_score[index]),
                delta=float(delta[index]),
                z=float(z[index]),
                delta_z=float(delta_z[index]),
                intensity=min(
                    1.0, float(evidence[index]) / MAX_INTENSITY_EVIDENCE
                ),
            )
            for index, name in enumerate(result.names)
        )
        selected_index = (
            select_voice_emotion(evidence, result.names, previous_name)
            if total > 0
            else None
        )
        component_tuple = (
            (signals[selected_index],) if selected_index is not None else ()
        )
        previous_name = (
            component_tuple[0].name if component_tuple else None
        )
        direction = compile_direction(
            component_tuple[0] if component_tuple else None,
            delivery_mode=delivery_mode,
        )
        spoken_text = re.sub(r"[*_~`]", "", text)
        segments.append(f"{direction} {spoken_text}" if direction else spoken_text)
        decisions.append(
            SpeechDecision(
                text=text,
                start_token=start,
                end_token=end,
                signals=signals,
                components=component_tuple,
                direction=direction,
            )
        )

    return " ".join(segments), decisions


def print_transfer(decisions: list[SpeechDecision]) -> None:
    print("\n=== Weight-to-voice transfer ===")
    print("Normalized evidence shares are diagnostics, not probabilities.")
    for index, decision in enumerate(decisions, start=1):
        print(f"\n[{index:02d}] tokens {decision.start_token}:{decision.end_token} {decision.text!r}")
        if not decision.components:
            print("     neutral     no meaningful evidence")
        for component in decision.components:
            filled = round(component.weight * 16)
            bar = "█" * filled + "░" * (16 - filled)
            print(
                f"     {component.name:<10} {bar} {component.weight:>5.1%}  "
                f"raw {component.raw:+.4f}  z {component.z:+.2f}  "
                f"Δz {component.delta_z:+.2f}  "
                f"intensity {component.intensity:.2f}"
            )
        print(f"     ElevenLabs  {decision.direction or '[neutral]'}")


def resolve_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Set ELEVENLABS_API_KEY or run in a terminal to enter it securely."
        )
    key = getpass.getpass("ElevenLabs API key (not stored): ").strip()
    if not key:
        raise RuntimeError("An ElevenLabs API key is required with --speak.")
    return key


class SpeechClient:
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str,
        output_format: str,
        stability: float,
    ) -> None:
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.stability = stability
        self.http = httpx.Client(
            base_url=ELEVENLABS_API,
            headers={"xi-api-key": api_key},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def close(self) -> None:
        self.http.close()

    def iter_audio(self, text: str) -> Iterator[bytes]:
        voice_settings = (
            {"stability": self.stability}
            if self.model_id == "eleven_v3"
            else {
                "stability": self.stability,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
                "speed": 1.0,
            }
        )
        with self.http.stream(
            "POST",
            f"/v1/text-to-speech/{self.voice_id}/stream",
            params={"output_format": self.output_format},
            json={
                "text": text,
                "model_id": self.model_id,
                "voice_settings": voice_settings,
            },
        ) as response:
            response.raise_for_status()
            yield from response.iter_bytes(chunk_size=4096)

    def transcribe(self, audio: bytes, *, content_type: str) -> dict:
        """Transcribe one in-memory recording with ElevenLabs Scribe v2."""
        extension = {
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/mp4": "m4a",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/webm": "webm",
            "video/webm": "webm",
        }.get(content_type, "audio")
        response = self.http.post(
            "/v1/speech-to-text",
            data={
                "model_id": "scribe_v2",
                "timestamps_granularity": "word",
                "tag_audio_events": "true",
            },
            files={"file": (f"utterance.{extension}", audio, content_type)},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Scribe v2 returned an invalid response")
        transcript = payload.get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            raise RuntimeError("Scribe v2 returned an empty transcript")
        return payload

    def synthesize_with_timestamps(
        self, text: str
    ) -> tuple[bytes, dict | None]:
        """Generate one Eleven v3 MP3 and its original-text alignment."""
        voice_settings = (
            {"stability": self.stability}
            if self.model_id == "eleven_v3"
            else {
                "stability": self.stability,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
                "speed": 1.0,
            }
        )
        response = self.http.post(
            f"/v1/text-to-speech/{self.voice_id}/with-timestamps",
            params={"output_format": self.output_format},
            json={
                "text": text,
                "model_id": self.model_id,
                "voice_settings": voice_settings,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("ElevenLabs returned an invalid response")
        encoded_audio = payload.get("audio_base64")
        if not isinstance(encoded_audio, str):
            raise RuntimeError("ElevenLabs returned no audio")
        try:
            audio = base64.b64decode(encoded_audio, validate=True)
        except ValueError as error:
            raise RuntimeError("ElevenLabs returned invalid audio") from error
        if not audio:
            raise RuntimeError("ElevenLabs returned empty audio")
        alignment = payload.get("alignment") or payload.get("normalized_alignment")
        return audio, alignment if isinstance(alignment, dict) else None

    def speak(
        self, text: str, *, save_path: Path | None
    ) -> tuple[float, float, Path | None]:
        ffplay = shutil.which("ffplay")
        if ffplay is None and save_path is None:
            save_path = Path("/tmp/nullhack-emotion-speech.mp3")

        player = None
        playback_enabled = ffplay is not None
        if ffplay is not None:
            player = subprocess.Popen(
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", "-i", "pipe:0"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        audio_file = None
        started = perf_counter()
        first_audio: float | None = None
        try:
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                audio_file = save_path.open("wb")
            for chunk in self.iter_audio(text):
                if not chunk:
                    continue
                if first_audio is None:
                    first_audio = perf_counter() - started
                if audio_file is not None:
                    audio_file.write(chunk)
                if playback_enabled and player is not None and player.stdin is not None:
                    try:
                        player.stdin.write(chunk)
                        player.stdin.flush()
                    except BrokenPipeError:
                        playback_enabled = False
        finally:
            if audio_file is not None:
                audio_file.close()
            if player is not None:
                if player.stdin is not None and not player.stdin.closed:
                    player.stdin.close()
                player.wait(timeout=30)

        total = perf_counter() - started
        return first_audio or total, total, save_path


def run_prompt(
    prompt: str,
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    names: list[str],
    vectors: torch.Tensor,
    neutral_mean: torch.Tensor,
    neutral_std: torch.Tensor,
    speech: SpeechClient | None,
) -> None:
    result = analyze_prompt(
        prompt,
        model=model,
        tokenizer=tokenizer,
        names=names,
        vectors=vectors,
        neutral_mean=neutral_mean,
        neutral_std=neutral_std,
        max_new_tokens=args.max_new_tokens,
    )
    print_trace(result, top_k=args.top_k, timings=args.timings)

    if speech is None:
        return
    tagged_text, decisions = build_tagged_speech(
        result,
        tokenizer,
        delivery_mode=args.delivery_mode,
        phrase_tokens=args.phrase_tokens,
    )
    print_transfer(decisions)
    print("\n=== Expressive speech ===")
    print(tagged_text)
    first_audio, total, saved = speech.speak(tagged_text, save_path=args.save_audio)
    if args.timings:
        print(f"Speech: first audio {first_audio:.2f}s · total {total:.2f}s")
    if saved is not None:
        print(f"Audio saved to {saved}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = parse_args()
    if args.max_new_tokens < 1 or args.top_k < 1:
        raise SystemExit("token counts and top-k values must be positive")
    if not 0 <= args.tts_stability <= 1:
        raise SystemExit("--tts-stability must be between 0 and 1")
    if args.phrase_tokens < 3:
        raise SystemExit("--phrase-tokens must be at least 3")
    if not args.interactive and not args.prompt:
        raise SystemExit("Provide a prompt or use --interactive")

    (
        model,
        tokenizer,
        names,
        vectors,
        neutral_mean,
        neutral_std,
        load_seconds,
    ) = load_runtime()
    if args.timings:
        print(f"Model load: {load_seconds:.2f}s")

    speech = None
    if args.speak:
        speech = SpeechClient(
            api_key=resolve_api_key(),
            voice_id=args.voice_id,
            model_id=args.tts_model,
            output_format=args.output_format,
            stability=args.tts_stability,
        )

    try:
        if args.prompt:
            run_prompt(
                args.prompt,
                args=args,
                model=model,
                tokenizer=tokenizer,
                names=names,
                vectors=vectors,
                neutral_mean=neutral_mean,
                neutral_std=neutral_std,
                speech=speech,
            )

        if args.interactive:
            print("\nGemma is ready. Enter a prompt; use /quit or Ctrl-D to exit.")
            while True:
                try:
                    prompt = input("\nemotion> ").strip()
                except EOFError:
                    print()
                    break
                if prompt == "/quit":
                    break
                if not prompt:
                    continue
                run_prompt(
                    prompt,
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    names=names,
                    vectors=vectors,
                    neutral_mean=neutral_mean,
                    neutral_std=neutral_std,
                    speech=speech,
                )
    finally:
        if speech is not None:
            speech.close()


if __name__ == "__main__":
    main()
