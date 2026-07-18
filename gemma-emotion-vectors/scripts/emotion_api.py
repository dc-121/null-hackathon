"""Shared emotion contract and pure helpers for the Emotion Mirror API.

The public/control surface intentionally contains only the five emotions shared
by the camera, language, voice and crowd renderers.  Gemma may still calculate
all nine vector diagnostics; callers keep those separately from these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
from typing import Any, Mapping, Sequence

import torch


SHARED_EMOTIONS = ("happy", "sad", "angry", "afraid", "surprised")

# Phrase evidence is directional rather than probabilistic. The UI supports
# the shared five, so phrase control selects inside that contract while still
# exposing how much all-nine evidence landed there.
PHRASE_EVIDENCE_CONTRAST = 1.8
MIN_EXPRESSION_INTENSITY = 0.45
FULL_EXPRESSION_EVIDENCE = 4.5
CONTINUATION_EVIDENCE_RATIO = 0.55
LANGUAGE_CONFIDENCE_CAP = 0.35
MIN_PROMPT_MODALITY_CONFIDENCE = 0.05
MIN_DOMINANCE_MARGIN = 0.05

RESPONSE_STRATEGIES = {
    "happy": (
        "celebrate",
        "Share the positive energy and recognize what went well without sounding exaggerated.",
    ),
    "sad": (
        "support",
        "Acknowledge the weight in what they said and make room for them to continue without diagnosing them.",
    ),
    "angry": (
        "de-escalate",
        "Validate the frustration, stay steady, and help move the conversation forward without mirroring hostility.",
    ),
    "afraid": (
        "reassure",
        "Respond calmly, ground the next step, and reassure without minimizing the concern.",
    ),
    "surprised": (
        "orient",
        "Acknowledge the surprise and help the person make sense of what just happened.",
    ),
}

STRATEGY_FIRST_MOVES = {
    "celebrate": "Open by sharing their positive momentum, then ask what feels most exciting or meaningful.",
    "support": "Open by gently acknowledging that the moment may feel heavier than the words alone suggest, then make room for them to continue.",
    "de-escalate": "Open by recognizing the frustration without mirroring its heat, then offer one calm way forward.",
    "reassure": "Open with a steady grounding sentence, then offer one concrete next step without minimizing the concern.",
    "orient": "Open with brief, genuine surprise, then ask one clarifying question that helps make sense of what happened.",
    "stay-curious": "Open with a tentative observation or question instead of assuming how they feel.",
}

STRATEGY_DELIVERY_TAGS = {
    "support": "warmly",
    "de-escalate": "calm",
    "reassure": "gentle",
    "stay-curious": "curious",
}

# HSEmotion ONNX's eight-class model uses title-cased noun labels.  Contempt,
# disgust and neutral are deliberately not folded into a different emotion: the
# omitted probability is reported as retained_mass instead of being hidden.
FACE_LABEL_MAP = {
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "afraid",
    "surprise": "surprised",
}

INPUT_LABEL_MAP = {
    **FACE_LABEL_MAP,
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "afraid": "afraid",
    "fearful": "afraid",
    "surprised": "surprised",
}

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def empty_scores() -> dict[str, float]:
    return {name: 0.0 for name in SHARED_EMOTIONS}


def canonical_scores(
    values: Mapping[str, float],
    *,
    aliases: Mapping[str, str] = INPUT_LABEL_MAP,
    normalize: bool = True,
) -> tuple[dict[str, float], float]:
    """Map non-negative scores to the five-emotion contract.

    Returns ``(scores, retained_mass)``.  ``retained_mass`` is measured before
    renormalization, which is useful for an eight-class face model where
    neutral/contempt/disgust are intentionally left out.
    """

    mapped = empty_scores()
    for raw_name, raw_value in values.items():
        name = aliases.get(str(raw_name).strip().lower())
        if name is None:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("emotion scores must be finite and non-negative")
        mapped[name] += value

    retained_mass = sum(mapped.values())
    if normalize and retained_mass > 0:
        mapped = {name: value / retained_mass for name, value in mapped.items()}
    return mapped, retained_mass


def validate_input_scores(values: Mapping[str, float]) -> dict[str, float]:
    unknown = {
        str(name).strip().lower()
        for name in values
        if str(name).strip().lower() not in INPUT_LABEL_MAP
    }
    if unknown:
        raise ValueError(
            "unsupported emotion labels: " + ", ".join(sorted(unknown))
        )
    scores, mass = canonical_scores(values)
    if mass <= 0:
        raise ValueError("emotion scores must contain positive evidence")
    return scores


def hsemotion_score_mapping(
    raw_scores: Any, idx_to_class: Mapping[int, str]
) -> dict[str, float]:
    """Attach HSEmotion's class labels to its NumPy probability vector."""

    values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if not isinstance(values, (list, tuple)):
        raise RuntimeError("HSEmotion returned an unexpected score format")
    if len(values) != len(idx_to_class):
        raise RuntimeError("HSEmotion returned an unexpected number of scores")
    try:
        return {
            str(idx_to_class[index]): float(value)
            for index, value in enumerate(values)
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("HSEmotion returned invalid class metadata") from error


@dataclass(frozen=True)
class Modality:
    name: str
    scores: Mapping[str, float]
    confidence: float
    retained_mass: float | None = None


@dataclass(frozen=True)
class FusionResult:
    scores: dict[str, float]
    dominant: str | None
    confidence: float
    modalities: dict[str, dict[str, Any]]


def fuse_modalities(modalities: Sequence[Modality]) -> FusionResult:
    """Confidence-weight emotion distributions and expose their agreement.

    The fused distribution is a conventional confidence-weighted mean.  The
    returned confidence also discounts conflicting modalities, so the prompt
    can correctly describe the result as uncertain sensor context.
    """

    prepared: list[
        tuple[str, dict[str, float], dict[str, float], float, float]
    ] = []
    for modality in modalities:
        confidence = float(modality.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("modality confidence must be between 0 and 1")
        if modality.retained_mass is None:
            public_scores = validate_input_scores(modality.scores)
            retained_mass = 1.0
        else:
            public_scores, measured_mass = canonical_scores(
                modality.scores, normalize=False
            )
            retained_mass = float(modality.retained_mass)
            if (
                not math.isfinite(retained_mass)
                or not 0 <= retained_mass <= 1
                or not math.isclose(measured_mass, retained_mass, abs_tol=1e-5)
                or (retained_mass == 0 and confidence > 0)
            ):
                raise ValueError("modality retained mass is invalid")
        conditional_scores = (
            {
                name: value / retained_mass
                for name, value in public_scores.items()
            }
            if retained_mass > 0
            else {name: 1.0 / len(SHARED_EMOTIONS) for name in SHARED_EMOTIONS}
        )
        prepared.append(
            (
                modality.name,
                public_scores,
                conditional_scores,
                confidence,
                retained_mass,
            )
        )

    if not prepared:
        raise ValueError("at least one modality is required")

    total_weight = sum(confidence for _, _, _, confidence, _ in prepared)
    if total_weight == 0:
        return FusionResult(
            scores={
                name: 1.0 / len(SHARED_EMOTIONS)
                for name in SHARED_EMOTIONS
            },
            dominant=None,
            confidence=0.0,
            modalities={
                name: {
                    "scores": public_scores,
                    "confidence": confidence,
                    "retained_mass": retained_mass,
                }
                for name, public_scores, _, confidence, retained_mass in prepared
            },
        )
    fused = {
        emotion: sum(
            scores[emotion] * confidence
            for _, _, scores, confidence, _ in prepared
        )
        / total_weight
        for emotion in SHARED_EMOTIONS
    }
    fused_total = sum(fused.values())
    fused = {name: value / fused_total for name, value in fused.items()}

    agreement = sum(
        confidence
        * (
            1.0
            - 0.5
            * sum(abs(scores[name] - fused[name]) for name in SHARED_EMOTIONS)
        )
        for _, _, scores, confidence, _ in prepared
    ) / total_weight
    independent_evidence = 1.0 - math.prod(
        1.0 - confidence for _, _, _, confidence, _ in prepared
    )
    confidence = max(0.0, min(1.0, independent_evidence * agreement))
    dominant = clear_dominant(fused)

    return FusionResult(
        scores=fused,
        dominant=dominant,
        confidence=confidence,
        modalities={
            name: {
                "scores": public_scores,
                "confidence": confidence,
                "retained_mass": retained_mass,
            }
            for name, public_scores, _, confidence, retained_mass in prepared
        },
    )


def clear_dominant(scores: dict[str, float]) -> str | None:
    """Return an argmax only when it is meaningfully ahead of the runner-up."""

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return (
        ranked[0][0]
        if ranked[0][1] - runner_up >= MIN_DOMINANCE_MARGIN
        else None
    )


def response_strategy(fusion: FusionResult) -> tuple[str, str]:
    """Choose a conservative default action, not an emotion imitation."""

    dominant = clear_dominant(fusion.scores)
    if dominant is None or fusion.confidence < 0.15:
        return (
            "stay-curious",
            "Keep the reply open and curious because the affect evidence is weak or mixed.",
        )
    language = fusion.modalities.get("language")
    language_dominant = None
    if (
        language
        and float(language["confidence"]) >= MIN_PROMPT_MODALITY_CONFIDENCE
    ):
        language_scores = canonical_scores(
            language["scores"], normalize=True
        )[0]
        language_dominant = clear_dominant(language_scores)
    sensor_dominants: set[str] = set()
    for source in ("face", "prosody"):
        modality = fusion.modalities.get(source)
        if (
            not modality
            or float(modality["confidence"])
            < MIN_PROMPT_MODALITY_CONFIDENCE
        ):
            continue
        scores = canonical_scores(modality["scores"], normalize=True)[0]
        source_dominant = clear_dominant(scores)
        if source_dominant is not None:
            sensor_dominants.add(source_dominant)
    if len(sensor_dominants) > 1 or (
        language_dominant in {"sad", "angry", "afraid"}
        and dominant in {"happy", "surprised"}
    ):
        return (
            "stay-curious",
            "The signals conflict. Check your understanding gently instead of asserting a mood or celebrating.",
        )
    return RESPONSE_STRATEGIES[dominant]


def explain_response_context(fusion: FusionResult) -> dict[str, Any]:
    """Describe nonverbal fusion weight and its directional context shift."""

    language = fusion.modalities.get("language")
    language_confidence = float(language["confidence"]) if language else 0.0
    language_scores = (
        canonical_scores(language["scores"], normalize=True)[0]
        if language and language_confidence > 0
        else empty_scores()
    )
    has_language_evidence = bool(
        language
        and language_confidence >= MIN_PROMPT_MODALITY_CONFIDENCE
        and sum(language_scores.values()) > 0
    )
    language_dominant = clear_dominant(language_scores)

    sensors: list[tuple[str, dict[str, float], float, str | None]] = []
    for name in ("face", "prosody"):
        modality = fusion.modalities.get(name)
        if not modality:
            continue
        confidence = float(modality["confidence"])
        if confidence < MIN_PROMPT_MODALITY_CONFIDENCE:
            continue
        scores = canonical_scores(modality["scores"], normalize=True)[0]
        if sum(scores.values()) <= 0:
            continue
        sensors.append((name, scores, confidence, clear_dominant(scores)))

    sensor_weight = sum(confidence for _, _, confidence, _ in sensors)
    total_weight = language_confidence + sensor_weight
    nonverbal_weight = sensor_weight / total_weight if total_weight > 0 else 0.0
    nonverbal_scores = empty_scores()
    if sensor_weight > 0:
        for emotion in SHARED_EMOTIONS:
            nonverbal_scores[emotion] = sum(
                scores[emotion] * confidence
                for _, scores, confidence, _ in sensors
            ) / sensor_weight
    nonverbal_dominant = clear_dominant(nonverbal_scores)
    nonverbal_shift = (
        0.5
        * sum(
            abs(fusion.scores[emotion] - language_scores[emotion])
            for emotion in SHARED_EMOTIONS
        )
        if has_language_evidence and sensors
        else None
    )
    source_dominants = {name: dominant for name, _, _, dominant in sensors}
    distinct_sensor_dominants = {
        dominant for dominant in source_dominants.values() if dominant is not None
    }
    if not sensors:
        effect = "words-only"
    elif len(distinct_sensor_dominants) > 1:
        effect = "mixed"
    elif not has_language_evidence:
        effect = "nonverbal-only"
    elif language_dominant is None:
        effect = "language-mixed"
    elif fusion.dominant != language_dominant:
        effect = "shifted"
    elif nonverbal_dominant == language_dominant:
        effect = "reinforced"
    else:
        effect = "adjusted"
    strategy, _ = response_strategy(fusion)
    return {
        "dominant": fusion.dominant,
        "confidence": fusion.confidence,
        "nonverbal_weight": max(0.0, min(1.0, nonverbal_weight)),
        "nonverbal_shift": (
            max(0.0, min(1.0, nonverbal_shift))
            if nonverbal_shift is not None
            else None
        ),
        "nonverbal_dominant": nonverbal_dominant,
        "language_dominant": language_dominant,
        "effect": effect,
        "sources": [name for name, _, _, _ in sensors],
        "source_dominants": source_dominants,
        "strategy": strategy,
    }


def shared_distribution(
    values: torch.Tensor, names: Sequence[str]
) -> tuple[dict[str, float], float, torch.Tensor]:
    """Softmax all vectors, then expose the shared subset without renormalizing."""

    missing = set(SHARED_EMOTIONS).difference(names)
    if missing:
        raise RuntimeError(
            "Gemma vector archive is missing: " + ", ".join(sorted(missing))
        )
    probabilities = torch.softmax(values, dim=0)
    scores = {
        name: float(probabilities[names.index(name)])
        for name in SHARED_EMOTIONS
    }
    retained_mass = sum(scores.values())
    if 1 < retained_mass <= 1.00001:
        scores = {
            name: value / retained_mass for name, value in scores.items()
        }
        retained_mass = 1.0
    return scores, retained_mass, probabilities


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def score_language_emotion(
    text: str,
    *,
    model,
    tokenizer,
    names: list[str],
    vectors: torch.Tensor,
    neutral_mean: torch.Tensor,
    neutral_std: torch.Tensor,
) -> tuple[dict[str, float], float, dict[str, dict[str, float]]]:
    """Score transcript tokens against every Gemma emotion vector.

    The five-way softmax is returned to the public fusion pipeline.  Raw cosine
    and calibrated z diagnostics for all vectors are returned separately so the
    runtime can retain the full nine-vector view without expanding the control
    taxonomy.
    """

    encoded = tokenizer(
        text.strip(),
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=1_024,
    ).to(_model_device(model))
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().float().cpu()

    language_model = getattr(model.model, "language_model", model.model)
    # Kept local to avoid a circular import from emotion_trace.
    target_layer = 28
    handle = language_model.layers[target_layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(**encoded)
    finally:
        handle.remove()

    hidden = captured["hidden"][0]
    attention = encoded.get("attention_mask")
    if attention is not None:
        mask = attention[0].detach().bool().cpu()
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        if special_ids:
            input_ids = encoded["input_ids"][0].detach().cpu()
            special_mask = torch.tensor(
                [int(token_id) not in special_ids for token_id in input_ids],
                dtype=torch.bool,
            )
            mask &= special_mask
        hidden = hidden[mask]
    normalized_hidden = torch.nn.functional.normalize(hidden, dim=-1)
    raw = (normalized_hidden @ vectors.T).mean(dim=0)
    z = (raw - neutral_mean) / neutral_std

    scores, retained_mass, probabilities = shared_distribution(z, names)
    all_diagnostics = {
        name: {
            "raw": float(raw[index]),
            "z": float(z[index]),
            "probability": float(probabilities[index]),
        }
        for index, name in enumerate(names)
    }
    all_diagnostics["_shared"] = {
        "retained_mass": retained_mass,
        "confidence": max(scores.values()),
    }

    # This is the strongest shared probability in the full nine-way softmax.
    # It falls below 1/9 when excluded vectors dominate instead of being
    # inflated to the old five-way minimum of 20%.
    confidence = max(scores.values())
    return scores, confidence, all_diagnostics


def response_prompt(transcript: str, fusion: FusionResult) -> str:
    ranked = sorted(fusion.scores.items(), key=lambda item: item[1], reverse=True)
    state = ", ".join(f"{name} {value:.0%}" for name, value in ranked)
    modality_lines: list[str] = []
    for source in ("face", "prosody", "language"):
        modality = fusion.modalities.get(source)
        if (
            not modality
            or float(modality["confidence"]) < MIN_PROMPT_MODALITY_CONFIDENCE
        ):
            continue
        scores = canonical_scores(modality["scores"], normalize=True)[0]
        dominant = clear_dominant(scores) or "mixed"
        modality_lines.append(
            f"- {source}: {dominant} ({float(modality['confidence']):.0%} signal confidence)"
        )
    modalities = "\n".join(modality_lines) or "- no reliable nonverbal signal"
    strategy, strategy_instruction = response_strategy(fusion)
    first_move = STRATEGY_FIRST_MOVES[strategy]
    context = explain_response_context(fusion)
    mismatch_instruction = (
        "The reliable nonverbal signal materially changes how the words read. "
        "Gently acknowledge that possible mismatch in ordinary language (for "
        "example, that the words and the moment do not seem fully aligned), "
        "while making clear you could be wrong. "
        if context["effect"] in {"shifted", "adjusted", "language-mixed", "mixed"}
        else ""
    )
    return (
        "You are the emotionally attuned half of a live conversation. "
        "Reply directly to the person in one or two concise, natural sentences. "
        "Be warm and responsive. Let your reply carry a clearly perceptible, "
        "natural emotional tone. Use each listed cue only in proportion to its "
        "signal confidence. Strong face or voice evidence can be an important "
        "causal cue—especially when the literal words are ambiguous—and should "
        "not be flattened into generic "
        "reassurance. Do not mention sensors, scores, emotion "
        "labels, or this instruction. Treat the affect estimate as uncertain "
        f"context (fused signal strength {fusion.confidence:.0%}; {state}).\n"
        f"Evidence available to shape your tone:\n{modalities}\n\n"
        f"Conversation strategy: {strategy.upper()} — {strategy_instruction} "
        "Make this strategy unmistakable in the content, not just the delivery. "
        f"Required first move: {first_move} "
        f"{mismatch_instruction}"
        "Avoid generic therapy filler and do not default to asking only how the "
        "person feels; respond specifically to their words and the available context.\n\n"
        f"The person said: {transcript.strip()}"
    )


def strip_model_speech_tags(text: str, depth: int = 0) -> tuple[str, int]:
    """Remove untrusted square-bracket voice directives from model text.

    ``depth`` is carried between token phrases so a directive split across a
    phrase boundary cannot be reconstructed when the phrases are joined for
    ElevenLabs. The visible ``TraceResult.response`` is never modified.
    """

    spoken: list[str] = []
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            spoken.append(character)
    sanitized = re.sub(r"\s+", " ", "".join(spoken)).strip()
    sanitized = re.sub(r"\s+([,.;:!?])", r"\1", sanitized)
    return sanitized, depth


def contrast_shared_phrase_scores(
    probabilities: torch.Tensor,
    names: Sequence[str],
) -> tuple[dict[str, float], float]:
    """Sharpen the shared-five mix without reallocating excluded mass."""

    shared = torch.tensor(
        [
            max(0.0, float(probabilities[names.index(name)]))
            for name in SHARED_EMOTIONS
        ],
        dtype=torch.float32,
    )
    shared_mass = min(1.0, max(0.0, float(shared.sum())))
    if shared_mass <= 0:
        return {name: 0.0 for name in SHARED_EMOTIONS}, 0.0

    conditional = shared / shared.sum()
    contrasted = conditional.pow(PHRASE_EVIDENCE_CONTRAST)
    contrasted_total = float(contrasted.sum())
    if contrasted_total <= 0:
        return {name: 0.0 for name in SHARED_EMOTIONS}, shared_mass
    contrasted = contrasted / contrasted_total * shared_mass
    return {
        name: float(contrasted[index])
        for index, name in enumerate(SHARED_EMOTIONS)
    }, shared_mass


def phrase_expression_intensity(
    evidence: float,
    *,
    minimum_evidence: float,
) -> float:
    """Calibrate validated shared evidence into visible delivery strength."""

    if evidence < minimum_evidence:
        return 0.0
    span = max(1e-6, FULL_EXPRESSION_EVIDENCE - minimum_evidence)
    progress = min(1.0, max(0.0, (evidence - minimum_evidence) / span))
    return min(
        1.0,
        MIN_EXPRESSION_INTENSITY
        + (1.0 - MIN_EXPRESSION_INTENSITY) * progress**0.75,
    )


def shared_phrase_plan(
    result,
    tokenizer,
    *,
    phrase_tokens: int = 10,
    strategy: str | None = None,
):
    """Create v3 directions and five-way phrase scores from a TraceResult.

    Returns ``(tagged_text, phrase_dicts, text_spans)``.  Text spans point to
    each phrase's spoken characters inside ``tagged_text`` and are used to map
    ElevenLabs character alignments back to phrases.
    """

    try:
        from emotion_trace import (
            EmotionComponent,
            MAX_INTENSITY_EVIDENCE,
            MIN_VOICE_EVIDENCE,
            compile_direction,
            emotion_evidence,
            phrase_spans,
        )
    except ModuleNotFoundError:  # imported as scripts.emotion_api in tests
        from scripts.emotion_trace import (
            EmotionComponent,
            MAX_INTENSITY_EVIDENCE,
            MIN_VOICE_EVIDENCE,
            compile_direction,
            emotion_evidence,
            phrase_spans,
        )

    missing = set(SHARED_EMOTIONS).difference(result.names)
    if missing:
        raise RuntimeError(
            "Gemma vector archive is missing: " + ", ".join(sorted(missing))
        )
    shared_indices = [result.names.index(name) for name in SHARED_EMOTIONS]
    baseline = result.scores[0]
    tagged_text = ""
    phrases: list[dict[str, Any]] = []
    text_spans: list[tuple[int, int]] = []
    model_tag_depth = 0
    previous_emotion: str | None = None
    previous_intensity = 0.0
    previous_text = ""

    for start, end in phrase_spans(
        tokenizer, result.response_ids, max_tokens=phrase_tokens
    ):
        text = tokenizer.decode(
            result.response_ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        text = re.sub(r"[*_~`]", "", text).strip()
        text, model_tag_depth = strip_model_speech_tags(text, model_tag_depth)
        if not text:
            continue

        mean_score = result.scores[start + 1 : end + 1].mean(dim=0)
        delta, z, delta_z, evidence = emotion_evidence(
            mean_score,
            baseline,
            result.neutral_mean,
            result.neutral_std,
        )
        total_evidence = float(evidence.sum())
        if total_evidence > 0:
            probabilities = evidence / evidence.sum()
        else:
            probabilities = torch.softmax(z, dim=0)
        scores, shared_mass = contrast_shared_phrase_scores(
            probabilities, result.names
        )

        # Select the strongest independently supported channel inside the
        # product's shared-five contract. A stronger excluded vector remains
        # visible through shared_mass, but no longer erases valid shared
        # evidence and turns an expressive phrase into emotion:null.
        winner_index = max(
            shared_indices,
            key=lambda index: float(evidence[index]),
        )
        winner_name = result.names[winner_index]
        selected_evidence = float(evidence[winner_index])
        has_voice_evidence = selected_evidence >= MIN_VOICE_EVIDENCE
        emotion = winner_name if has_voice_evidence else None
        intensity = phrase_expression_intensity(
            selected_evidence,
            minimum_evidence=MIN_VOICE_EVIDENCE,
        )
        continues_sentence = bool(previous_text) and not re.search(
            r"[.!?][\"')\]]*$", previous_text
        )
        if (
            emotion is None
            and previous_emotion
            and continues_sentence
            and result.names[winner_index] == previous_emotion
        ):
            previous_index = result.names.index(previous_emotion)
            continuation_evidence = float(evidence[previous_index])
            continuation_floor = MIN_VOICE_EVIDENCE * CONTINUATION_EVIDENCE_RATIO
            if continuation_evidence >= continuation_floor:
                emotion = previous_emotion
                winner_index = previous_index
                winner_name = previous_emotion
                selected_evidence = continuation_evidence
                intensity = max(
                    MIN_EXPRESSION_INTENSITY,
                    min(
                        previous_intensity * 0.9,
                        phrase_expression_intensity(
                            continuation_evidence,
                            minimum_evidence=continuation_floor,
                        ),
                    ),
                )
        component = (
            EmotionComponent(
                name=emotion,
                weight=scores[emotion],
                evidence=selected_evidence,
                raw=float(mean_score[winner_index]),
                delta=float(delta[winner_index]),
                z=float(z[winner_index]),
                delta_z=float(delta_z[winner_index]),
                intensity=intensity,
            )
            if emotion is not None
            else None
        )
        if component is not None and strategy in STRATEGY_DELIVERY_TAGS:
            direction = f"[{STRATEGY_DELIVERY_TAGS[strategy]}]"
        else:
            direction = compile_direction(component, delivery_mode="raw")
        segment = f"{direction} {text}" if direction else text
        if tagged_text:
            tagged_text += " "
        segment_start = len(tagged_text)
        tagged_text += segment
        text_start = segment_start + len(segment) - len(text)
        text_spans.append((text_start, text_start + len(text)))
        phrases.append(
            {
                "text": text,
                "emotion": emotion,
                "scores": scores,
                "intensity": intensity,
                "evidence": max(0.0, selected_evidence),
                "shared_mass": shared_mass,
                "direction": direction,
                "start_seconds": None,
                "end_seconds": None,
            }
        )
        previous_emotion = emotion
        previous_intensity = intensity
        previous_text = text

    return tagged_text, phrases, text_spans


def add_phrase_timings(
    phrases: list[dict[str, Any]],
    text_spans: Sequence[tuple[int, int]],
    alignment: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach ElevenLabs original-text character alignment to phrase records."""

    if not alignment:
        return phrases
    characters = alignment.get("characters")
    starts = alignment.get("character_start_times_seconds")
    ends = alignment.get("character_end_times_seconds")
    if not (
        isinstance(characters, list)
        and isinstance(starts, list)
        and isinstance(ends, list)
        and len(characters) == len(starts) == len(ends)
    ):
        return phrases

    aligned_text = "".join(str(character) for character in characters)
    # ElevenLabs documents this alignment as one entry per original input
    # character. If a provider normalization ever changes that, locate each
    # phrase sequentially instead of silently assigning wrong times.
    cursor = 0
    for phrase, (expected_start, expected_end) in zip(
        phrases, text_spans, strict=True
    ):
        text = phrase["text"]
        start = expected_start
        end = expected_end
        if aligned_text[start:end] != text:
            found = aligned_text.find(text, cursor)
            if found < 0:
                continue
            start, end = found, found + len(text)
        if start >= len(starts) or end <= start or end > len(ends):
            continue
        phrase["start_seconds"] = float(starts[start])
        phrase["end_seconds"] = float(ends[end - 1])
        cursor = end
    return phrases


class FaceInferenceUnavailable(RuntimeError):
    pass


class FaceEmotionAnalyzer:
    """Lazy, thread-safe OpenCV + Apache-2.0 HSEmotion ONNX inference."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._cv2 = None
        self._detector = None
        self._recognizer = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._recognizer is not None:
            return
        if self._load_error is not None:
            raise FaceInferenceUnavailable(self._load_error)
        with self._load_lock:
            if self._recognizer is not None:
                return
            if self._load_error is not None:
                raise FaceInferenceUnavailable(self._load_error)
            try:
                import cv2  # type: ignore[import-not-found]
                from hsemotion_onnx.facial_emotions import (  # type: ignore[import-not-found]
                    HSEmotionRecognizer,
                )

                cascade_path = (
                    str(cv2.data.haarcascades)
                    + "haarcascade_frontalface_default.xml"
                )
                detector = cv2.CascadeClassifier(cascade_path)
                if detector.empty():
                    raise RuntimeError("OpenCV's bundled face detector did not load")
                # HSEmotion ONNX downloads its published model on first use. No
                # YuNet artifact is downloaded by this project; the detector is
                # OpenCV's packaged cascade, so there is no unchecked download.
                recognizer = HSEmotionRecognizer(
                    model_name="enet_b0_8_best_afew"
                )
            except Exception as error:
                self._load_error = (
                    "install the face dependencies and ensure the HSEmotion "
                    f"model is available ({type(error).__name__})"
                )
                raise FaceInferenceUnavailable(self._load_error) from error
            self._cv2 = cv2
            self._detector = detector
            self._recognizer = recognizer

    def analyze(self, jpeg: bytes) -> dict[str, Any]:
        if not jpeg:
            raise ValueError("JPEG body is empty")
        if len(jpeg) > MAX_IMAGE_BYTES:
            raise ValueError("JPEG body exceeds 8 MiB")
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("body is not a JPEG image")
        self._load()
        assert self._cv2 is not None
        assert self._detector is not None
        assert self._recognizer is not None

        cv2 = self._cv2
        # numpy is already a transitive dependency of the Gemma runtime, but it
        # stays inside the lazy face path so missing face extras fail cleanly.
        import numpy as np

        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("body is not a decodable JPEG image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        with self._inference_lock:
            faces = self._detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )
            if len(faces) == 0:
                return {
                    "detected": False,
                    "scores": empty_scores(),
                    "retained_mass": 0.0,
                    "confidence": 0.0,
                    "dominant": None,
                }
            x, y, width, height = max(
                faces, key=lambda face: int(face[2]) * int(face[3])
            )
            pad_x = int(width * 0.08)
            pad_y = int(height * 0.08)
            x0 = max(0, int(x) - pad_x)
            y0 = max(0, int(y) - pad_y)
            x1 = min(image.shape[1], int(x + width) + pad_x)
            y1 = min(image.shape[0], int(y + height) + pad_y)
            face = image[y0:y1, x0:x1]
            _label, raw_scores = self._recognizer.predict_emotions(
                face, logits=False
            )

        raw_scores = hsemotion_score_mapping(
            raw_scores, self._recognizer.idx_to_class
        )
        scores, retained_mass = canonical_scores(
            raw_scores, aliases=FACE_LABEL_MAP
        )
        if retained_mass <= 0:
            raise RuntimeError("HSEmotion returned no shared-emotion evidence")
        dominant = max(SHARED_EMOTIONS, key=scores.__getitem__)
        ordered = sorted(scores.values(), reverse=True)
        separation = ordered[0] - ordered[1]
        confidence = min(
            1.0,
            max(0.0, retained_mass * (0.5 + 0.5 * separation)),
        )
        return {
            "detected": True,
            "scores": scores,
            "retained_mass": retained_mass,
            "confidence": confidence,
            "dominant": dominant,
        }
