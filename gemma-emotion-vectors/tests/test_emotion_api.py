import base64
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
import torch

from scripts.emotion_api import (
    FACE_LABEL_MAP,
    Modality,
    SHARED_EMOTIONS,
    add_phrase_timings,
    canonical_scores,
    explain_response_context,
    fuse_modalities,
    hsemotion_score_mapping,
    response_prompt,
    shared_distribution,
    shared_phrase_plan,
)
from scripts.emotion_trace import SpeechClient, TraceResult
from scripts import emotion_web
from scripts.emotion_web import (
    API_VERSION,
    AnalyzeRequest,
    ConversationRequest,
    ConversationResponse,
)


class TaxonomyTests(unittest.TestCase):
    def test_hsemotion_vector_maps_only_the_shared_five(self) -> None:
        labels = {
            0: "Anger",
            1: "Contempt",
            2: "Disgust",
            3: "Fear",
            4: "Happiness",
            5: "Neutral",
            6: "Sadness",
            7: "Surprise",
        }
        vector = torch.tensor([0.10, 0.10, 0.10, 0.05, 0.20, 0.20, 0.15, 0.10])

        mapped = hsemotion_score_mapping(vector.numpy(), labels)
        scores, retained_mass = canonical_scores(mapped, aliases=FACE_LABEL_MAP)

        self.assertAlmostEqual(retained_mass, 0.60, places=6)
        self.assertEqual(tuple(scores), SHARED_EMOTIONS)
        self.assertAlmostEqual(sum(scores.values()), 1.0)
        self.assertAlmostEqual(scores["happy"], 1 / 3, places=6)
        self.assertNotIn("neutral", scores)
        self.assertNotIn("contempt", scores)

    def test_hsemotion_vector_and_metadata_must_match(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "number of scores"):
            hsemotion_score_mapping([0.2, 0.8], {0: "Happy"})

    def test_excluded_gemma_probability_is_not_reassigned(self) -> None:
        names = PhraseTests.names
        values = torch.zeros(len(names))
        values[names.index("loving")] = 10.0

        scores, retained_mass, all_probabilities = shared_distribution(
            values, names
        )

        self.assertEqual(tuple(scores), SHARED_EMOTIONS)
        self.assertAlmostEqual(sum(scores.values()), retained_mass)
        self.assertLess(retained_mass, 0.001)
        self.assertGreater(
            float(all_probabilities[names.index("loving")]),
            0.999,
        )

    def test_shared_float_rounding_never_exceeds_unit_mass(self) -> None:
        names = PhraseTests.names
        # This float32 softmax sums to 1.000000037 when converted element by
        # element to Python floats unless shared_distribution corrects it.
        values = torch.tensor(
            [
                0.0447275378,
                1.911239028,
                -100.2310333,
                -99.6540756,
                -98.6819687,
                0.3696370125,
                -99.6158905,
                0.2970382571,
                0.7472865582,
            ]
        )

        scores, retained_mass, _probabilities = shared_distribution(values, names)

        self.assertEqual(retained_mass, 1.0)
        self.assertAlmostEqual(sum(scores.values()), 1.0)
        fused = fuse_modalities(
            [
                Modality(
                    "language",
                    scores,
                    confidence=max(scores.values()),
                    retained_mass=retained_mass,
                )
            ]
        )
        self.assertIsNotNone(fused.dominant)


class FusionTests(unittest.TestCase):
    def test_fusion_is_confidence_weighted_and_reports_modalities(self) -> None:
        face = {name: 0.0 for name in SHARED_EMOTIONS}
        face["happy"] = 1.0
        language = {name: 0.0 for name in SHARED_EMOTIONS}
        language["sad"] = 1.0

        result = fuse_modalities(
            [
                Modality("face", face, 0.75),
                Modality("language", language, 0.25),
            ]
        )

        self.assertAlmostEqual(result.scores["happy"], 0.75)
        self.assertAlmostEqual(result.scores["sad"], 0.25)
        self.assertEqual(result.dominant, "happy")
        self.assertLess(result.confidence, 1.0)  # disagreement is explicit
        self.assertEqual(set(result.modalities), {"face", "language"})

    def test_partial_language_mass_remains_visible(self) -> None:
        partial = {name: 0.0 for name in SHARED_EMOTIONS}
        partial["sad"] = 0.05

        result = fuse_modalities(
            [
                Modality(
                    "language",
                    partial,
                    confidence=0.05,
                    retained_mass=0.05,
                )
            ]
        )

        language = result.modalities["language"]
        self.assertAlmostEqual(sum(language["scores"].values()), 0.05)
        self.assertAlmostEqual(language["retained_mass"], 0.05)
        self.assertAlmostEqual(language["confidence"], 0.05)

    def test_response_context_quantifies_nonverbal_fusion_contribution(self) -> None:
        happy = {name: 0.0 for name in SHARED_EMOTIONS}
        happy["happy"] = 1.0
        sad = {name: 0.0 for name in SHARED_EMOTIONS}
        sad["sad"] = 1.0
        fused = fuse_modalities(
            [
                Modality("language", happy, 0.35),
                Modality("face", sad, 0.8),
                Modality("prosody", sad, 0.4),
            ]
        )

        context = explain_response_context(fused)

        self.assertEqual(context["language_dominant"], "happy")
        self.assertEqual(context["nonverbal_dominant"], "sad")
        self.assertEqual(context["dominant"], "sad")
        self.assertEqual(context["effect"], "shifted")
        self.assertEqual(context["sources"], ["face", "prosody"])
        self.assertGreater(context["nonverbal_weight"], 0.7)
        self.assertGreater(context["nonverbal_shift"], 0.5)
        self.assertEqual(context["strategy"], "support")
        prompt = response_prompt("I guess it is over.", fused)
        self.assertIn("- face: sad", prompt)
        self.assertIn("- prosody: sad", prompt)
        self.assertIn("important causal cue", prompt)
        self.assertIn("Conversation strategy: SUPPORT", prompt)

    def test_low_confidence_conflict_is_not_promoted_in_the_prompt(self) -> None:
        happy = {name: 0.0 for name in SHARED_EMOTIONS}
        happy["happy"] = 1.0
        sad = {name: 0.0 for name in SHARED_EMOTIONS}
        sad["sad"] = 1.0
        fused = fuse_modalities(
            [
                Modality("language", happy, 0.35),
                Modality("face", sad, 0.001),
            ]
        )

        context = explain_response_context(fused)
        prompt = response_prompt("It is over.", fused)

        self.assertEqual(context["effect"], "words-only")
        self.assertEqual(context["sources"], [])
        self.assertNotIn("- face:", prompt)
        self.assertIn("only in proportion", prompt)

    def test_context_distinguishes_adjustment_mixed_and_nonverbal_only(self) -> None:
        happy = {name: 0.0 for name in SHARED_EMOTIONS}
        happy["happy"] = 1.0
        sad = {name: 0.0 for name in SHARED_EMOTIONS}
        sad["sad"] = 1.0
        empty = {name: 0.0 for name in SHARED_EMOTIONS}
        uniform = {name: 0.2 for name in SHARED_EMOTIONS}

        adjusted = explain_response_context(
            fuse_modalities(
                [
                    Modality("language", happy, 0.8),
                    Modality("face", sad, 0.1),
                ]
            )
        )
        mixed = explain_response_context(
            fuse_modalities(
                [
                    Modality("language", happy, 0.35),
                    Modality("face", happy, 0.8),
                    Modality("prosody", sad, 0.8),
                ]
            )
        )
        nonverbal_only = explain_response_context(
            fuse_modalities(
                [
                    Modality("language", empty, 0.0, retained_mass=0.0),
                    Modality("face", sad, 0.8),
                ]
            )
        )
        language_mixed = explain_response_context(
            fuse_modalities(
                [
                    Modality("language", uniform, 0.35),
                    Modality("face", sad, 0.8),
                ]
            )
        )

        self.assertEqual(adjusted["effect"], "adjusted")
        self.assertEqual(adjusted["dominant"], "happy")
        self.assertEqual(mixed["effect"], "mixed")
        self.assertEqual(mixed["source_dominants"], {"face": "happy", "prosody": "sad"})
        self.assertEqual(nonverbal_only["effect"], "nonverbal-only")
        self.assertEqual(nonverbal_only["nonverbal_weight"], 1.0)
        self.assertIsNone(nonverbal_only["nonverbal_shift"])
        self.assertEqual(language_mixed["effect"], "language-mixed")
        self.assertIsNone(language_mixed["language_dominant"])
        self.assertIsNotNone(language_mixed["nonverbal_shift"])

    def test_mixed_or_negative_conflict_uses_a_check_in_strategy(self) -> None:
        happy = {name: 0.0 for name in SHARED_EMOTIONS}
        happy["happy"] = 1.0
        sad = {name: 0.0 for name in SHARED_EMOTIONS}
        sad["sad"] = 1.0
        conflicted = fuse_modalities(
            [
                Modality("language", sad, 0.35),
                Modality("face", happy, 0.8),
            ]
        )

        context = explain_response_context(conflicted)
        prompt = response_prompt("My mother died today.", conflicted)

        self.assertEqual(context["effect"], "shifted")
        self.assertEqual(context["strategy"], "stay-curious")
        self.assertIn("Conversation strategy: STAY-CURIOUS", prompt)
        self.assertNotIn("Conversation strategy: CELEBRATE", prompt)

    def test_uniform_fusion_has_no_public_dominant(self) -> None:
        uniform = {name: 0.2 for name in SHARED_EMOTIONS}

        fused = fuse_modalities([Modality("language", uniform, 0.35)])
        context = explain_response_context(fused)

        self.assertIsNone(fused.dominant)
        self.assertIsNone(context["dominant"])
        self.assertEqual(context["strategy"], "stay-curious")

    def test_zero_shared_mass_is_uncertain_instead_of_reassigned(self) -> None:
        names = PhraseTests.names
        values = torch.full((len(names),), -200.0)
        values[names.index("loving")] = 200.0
        empty, retained_mass, _probabilities = shared_distribution(values, names)

        result = fuse_modalities(
            [
                Modality(
                    "language",
                    empty,
                    confidence=0.0,
                    retained_mass=retained_mass,
                )
            ]
        )

        self.assertEqual(retained_mass, 0.0)
        self.assertIsNone(result.dominant)
        self.assertEqual(result.confidence, 0.0)
        self.assertAlmostEqual(sum(result.scores.values()), 1.0)
        self.assertEqual(
            sum(result.modalities["language"]["scores"].values()),
            0.0,
        )


class FakeTokenizer:
    pieces = {
        1: "Bright",
        2: " day",
        3: ".",
        4: "Stay",
        5: " close",
        6: ".",
    }

    def decode(self, token_ids, **_kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.pieces[token_id] for token_id in token_ids)


class PhraseTests(unittest.TestCase):
    names = [
        "afraid",
        "angry",
        "calm",
        "desperate",
        "guilty",
        "happy",
        "loving",
        "sad",
        "surprised",
    ]

    def make_result(self) -> TraceResult:
        scores = torch.zeros((7, len(self.names)))
        # The excluded nine-way emotions dominate. Public voice control must
        # preserve that uncertainty instead of reassigning it to a shared label.
        for row in range(1, 4):
            scores[row, self.names.index("loving")] = 9.0
            scores[row, self.names.index("happy")] = 3.0
        for row in range(4, 7):
            scores[row, self.names.index("calm")] = 8.0
            scores[row, self.names.index("sad")] = 2.5
        return TraceResult(
            response="Bright day. Stay close.",
            response_ids=[1, 2, 3, 4, 5, 6],
            labels=[],
            names=self.names,
            scores=scores,
            neutral_mean=torch.zeros(len(self.names)),
            neutral_std=torch.ones(len(self.names)),
            generation_seconds=0.1,
            replay_seconds=0.1,
        )

    def test_strong_shared_runner_up_controls_the_five_emotion_surface(self) -> None:
        tagged, phrases, spans = shared_phrase_plan(
            self.make_result(), FakeTokenizer(), phrase_tokens=10
        )

        self.assertEqual(
            [phrase["emotion"] for phrase in phrases],
            ["happy", "sad"],
        )
        self.assertNotEqual(tagged, "Bright day. Stay close.")
        for phrase in phrases:
            self.assertEqual(tuple(phrase["scores"]), SHARED_EMOTIONS)
            self.assertLess(sum(phrase["scores"].values()), 1.0)
            self.assertNotIn("loving", phrase["scores"])
            self.assertNotIn("calm", phrase["scores"])
            self.assertTrue(phrase["direction"])
            self.assertGreaterEqual(phrase["intensity"], 0.45)
            self.assertGreaterEqual(phrase["evidence"], 1.25)
            self.assertAlmostEqual(
                phrase["shared_mass"],
                sum(phrase["scores"].values()),
            )

        alignment = {
            "characters": list(tagged),
            "character_start_times_seconds": [
                index / 10 for index in range(len(tagged))
            ],
            "character_end_times_seconds": [
                (index + 1) / 10 for index in range(len(tagged))
            ],
        }
        add_phrase_timings(phrases, spans, alignment)

        for phrase, (start, end) in zip(phrases, spans, strict=True):
            self.assertAlmostEqual(phrase["start_seconds"], start / 10)
            self.assertAlmostEqual(phrase["end_seconds"], end / 10)

    def test_shared_phrase_winner_can_control_voice(self) -> None:
        result = self.make_result()
        scores = torch.zeros_like(result.scores)
        for row in range(1, 4):
            scores[row, self.names.index("happy")] = 3.0
        for row in range(4, 7):
            scores[row, self.names.index("sad")] = 2.5
        result = TraceResult(
            response=result.response,
            response_ids=result.response_ids,
            labels=result.labels,
            names=result.names,
            scores=scores,
            neutral_mean=result.neutral_mean,
            neutral_std=result.neutral_std,
            generation_seconds=result.generation_seconds,
            replay_seconds=result.replay_seconds,
        )

        tagged, phrases, _spans = shared_phrase_plan(
            result, FakeTokenizer(), phrase_tokens=10
        )

        self.assertEqual(
            [phrase["emotion"] for phrase in phrases],
            ["happy", "sad"],
        )
        self.assertTrue(all(phrase["direction"] for phrase in phrases))
        self.assertNotEqual(tagged, result.response)

    def test_weak_shared_evidence_stays_untagged_and_uncertain(self) -> None:
        result = self.make_result()
        result = TraceResult(
            response=result.response,
            response_ids=result.response_ids,
            labels=result.labels,
            names=result.names,
            scores=torch.zeros_like(result.scores),
            neutral_mean=result.neutral_mean,
            neutral_std=result.neutral_std,
            generation_seconds=result.generation_seconds,
            replay_seconds=result.replay_seconds,
        )

        tagged, phrases, _spans = shared_phrase_plan(
            result, FakeTokenizer(), phrase_tokens=10
        )

        self.assertEqual(tagged, "Bright day. Stay close.")
        self.assertTrue(all(phrase["emotion"] is None for phrase in phrases))
        self.assertTrue(all(phrase["direction"] == "" for phrase in phrases))
        self.assertTrue(all(phrase["intensity"] == 0 for phrase in phrases))

    def test_weak_clause_continues_a_strong_sentence_tone(self) -> None:
        class ClauseTokenizer:
            pieces = {
                1: "I",
                2: " feel",
                3: ",",
                4: " still",
                5: " very",
                6: " low.",
            }

            def decode(self, token_ids, **_kwargs):
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(self.pieces[token_id] for token_id in token_ids)

        scores = torch.zeros((7, len(self.names)))
        scores[1:4, self.names.index("sad")] = 3.0
        scores[4:7, self.names.index("sad")] = 1.1
        result = TraceResult(
            response="I feel, still very low.",
            response_ids=[1, 2, 3, 4, 5, 6],
            labels=[],
            names=self.names,
            scores=scores,
            neutral_mean=torch.zeros(len(self.names)),
            neutral_std=torch.ones(len(self.names)),
            generation_seconds=0.1,
            replay_seconds=0.1,
        )

        tagged, phrases, _spans = shared_phrase_plan(
            result, ClauseTokenizer(), phrase_tokens=10
        )

        self.assertEqual([phrase["emotion"] for phrase in phrases], ["sad", "sad"])
        self.assertTrue(all(phrase["direction"] for phrase in phrases))
        self.assertGreaterEqual(phrases[1]["intensity"], 0.45)
        self.assertIn("still very low", tagged)

    def test_continuation_never_overrides_a_competing_shared_winner(self) -> None:
        class ClauseTokenizer:
            pieces = {
                1: "I",
                2: " feel",
                3: ",",
                4: " but",
                5: " something",
                6: " changed.",
            }

            def decode(self, token_ids, **_kwargs):
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(self.pieces[token_id] for token_id in token_ids)

        scores = torch.zeros((7, len(self.names)))
        scores[1:4, self.names.index("sad")] = 3.0
        scores[4:7, self.names.index("happy")] = 1.1
        scores[4:7, self.names.index("sad")] = 0.95
        result = TraceResult(
            response="I feel, but something changed.",
            response_ids=[1, 2, 3, 4, 5, 6],
            labels=[],
            names=self.names,
            scores=scores,
            neutral_mean=torch.zeros(len(self.names)),
            neutral_std=torch.ones(len(self.names)),
            generation_seconds=0.1,
            replay_seconds=0.1,
        )

        _tagged, phrases, _spans = shared_phrase_plan(
            result, ClauseTokenizer(), phrase_tokens=10
        )

        self.assertEqual(phrases[0]["emotion"], "sad")
        self.assertIsNone(phrases[1]["emotion"])
        self.assertGreater(
            phrases[1]["scores"]["happy"],
            phrases[1]["scores"]["sad"],
        )

    def test_deescalation_strategy_keeps_extreme_angry_voice_steady(self) -> None:
        original = self.make_result()
        scores = torch.zeros_like(original.scores)
        scores[1:, self.names.index("angry")] = 8.0
        result = TraceResult(
            response=original.response,
            response_ids=original.response_ids,
            labels=original.labels,
            names=original.names,
            scores=scores,
            neutral_mean=original.neutral_mean,
            neutral_std=original.neutral_std,
            generation_seconds=original.generation_seconds,
            replay_seconds=original.replay_seconds,
        )

        tagged, phrases, _spans = shared_phrase_plan(
            result,
            FakeTokenizer(),
            phrase_tokens=10,
            strategy="de-escalate",
        )

        self.assertTrue(all(phrase["emotion"] == "angry" for phrase in phrases))
        self.assertTrue(all(phrase["intensity"] == 1.0 for phrase in phrases))
        self.assertTrue(all(phrase["direction"] == "[calm]" for phrase in phrases))
        self.assertNotIn("furious", tagged)
        self.assertNotIn("shouts", tagged)

    def test_model_authored_voice_tag_is_not_sent_to_elevenlabs(self) -> None:
        class TaggedTokenizer:
            pieces = {
                1: "[",
                2: "sobbing",
                3: "]",
                4: " I",
                5: " am",
                6: " okay",
                7: ".",
            }

            def decode(self, token_ids, **_kwargs):
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(self.pieces[token_id] for token_id in token_ids)

        response = "[sobbing] I am okay."
        result = TraceResult(
            response=response,
            response_ids=[1, 2, 3, 4, 5, 6, 7],
            labels=[],
            names=self.names,
            scores=torch.zeros((8, len(self.names))),
            neutral_mean=torch.zeros(len(self.names)),
            neutral_std=torch.ones(len(self.names)),
            generation_seconds=0.1,
            replay_seconds=0.1,
        )

        tagged, phrases, _spans = shared_phrase_plan(
            result, TaggedTokenizer(), phrase_tokens=10
        )

        self.assertEqual(result.response, response)
        self.assertEqual(tagged, "I am okay.")
        self.assertEqual(phrases[0]["text"], "I am okay.")
        self.assertIsNone(phrases[0]["emotion"])
        self.assertEqual(phrases[0]["direction"], "")


class RequestValidationTests(unittest.TestCase):
    def test_typed_fallback_is_valid(self) -> None:
        request = ConversationRequest(
            api_version=API_VERSION,
            transcript="I am ready",
        )
        self.assertEqual(request.transcript, "I am ready")

    def test_audio_requires_content_type(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationRequest(
                api_version=API_VERSION,
                audio_base64=base64.b64encode(b"audio").decode(),
            )

    def test_transcript_and_audio_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationRequest(
                api_version=API_VERSION,
                transcript="hello",
                audio_base64=base64.b64encode(b"audio").decode(),
                audio_content_type="audio/webm",
            )

    def test_modality_scores_require_confidence(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationRequest(
                api_version=API_VERSION,
                transcript="hello",
                face_scores={"happy": 1.0},
            )

    def test_unknown_emotion_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationRequest(
                api_version=API_VERSION,
                transcript="hello",
                face_scores={"confused": 1.0},
                face_confidence=0.8,
            )

    def test_request_contract_rejects_unknown_fields_and_version_mismatch(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationRequest.model_validate(
                {
                    "api_version": API_VERSION,
                    "transcript": "hello",
                    "synthesize_speach": False,
                }
            )
        with self.assertRaises(ValidationError):
            AnalyzeRequest.model_validate({"prompt": "hello", "mystery": True})
        with self.assertRaises(ValidationError):
            ConversationRequest(
                transcript="hello",
                api_version="stale-client",
            )

    def test_api_version_is_visible_in_config_and_request_contract(self) -> None:
        self.assertEqual(emotion_web.config()["api_version"], API_VERSION)
        with self.assertRaises(ValidationError):
            ConversationRequest(transcript="hello")
        self.assertEqual(
            ConversationRequest(
                api_version=API_VERSION,
                transcript="hello",
            ).api_version,
            API_VERSION,
        )


class StartupPreflightTests(unittest.TestCase):
    def test_port_reservation_reports_bind_failure_and_closes_socket(self) -> None:
        server_socket = Mock()
        server_socket.bind.side_effect = OSError("address already in use")
        with patch.object(
            emotion_web.socket,
            "socket",
            return_value=server_socket,
        ):
            with self.assertRaisesRegex(SystemExit, "address already in use"):
                emotion_web.reserve_server_socket("127.0.0.1", 8766)

        server_socket.close.assert_called_once_with()

    def test_occupied_port_fails_before_model_load(self) -> None:
        args = SimpleNamespace(host="127.0.0.1", port=8766, no_open=True)
        with (
            patch.object(emotion_web, "parse_args", return_value=args),
            patch.object(
                emotion_web,
                "reserve_server_socket",
                side_effect=SystemExit("Cannot start server: occupied"),
            ),
            patch.object(emotion_web, "load_runtime") as load_runtime,
        ):
            with self.assertRaisesRegex(SystemExit, "Cannot start server"):
                emotion_web.main()

        load_runtime.assert_not_called()


class ElevenLabsClientTests(unittest.TestCase):
    def make_client(self) -> SpeechClient:
        client = SpeechClient(
            api_key="test-only",
            voice_id="voice",
            model_id="eleven_v3",
            output_format="mp3_22050_32",
            stability=0.25,
        )
        client.http.close()
        client.http = Mock()
        return client

    def test_scribe_uses_v2_multipart_without_network(self) -> None:
        response = Mock()
        response.json.return_value = {"text": "hello", "words": []}
        client = self.make_client()
        client.http.post.return_value = response

        payload = client.transcribe(b"fake-audio", content_type="audio/webm")

        self.assertEqual(payload["text"], "hello")
        args, kwargs = client.http.post.call_args
        self.assertEqual(args[0], "/v1/speech-to-text")
        self.assertEqual(kwargs["data"]["model_id"], "scribe_v2")
        self.assertEqual(kwargs["files"]["file"][2], "audio/webm")
        response.raise_for_status.assert_called_once_with()

    def test_v3_speech_uses_timestamp_endpoint_without_network(self) -> None:
        alignment = {
            "characters": ["H", "i"],
            "character_start_times_seconds": [0.0, 0.1],
            "character_end_times_seconds": [0.1, 0.2],
        }
        response = Mock()
        response.json.return_value = {
            "audio_base64": base64.b64encode(b"ID3audio").decode(),
            "alignment": alignment,
        }
        client = self.make_client()
        client.http.post.return_value = response

        audio, returned_alignment = client.synthesize_with_timestamps("Hi")

        self.assertEqual(audio, b"ID3audio")
        self.assertEqual(returned_alignment, alignment)
        args, kwargs = client.http.post.call_args
        self.assertEqual(
            args[0],
            "/v1/text-to-speech/voice/with-timestamps",
        )
        self.assertEqual(kwargs["json"]["model_id"], "eleven_v3")
        self.assertEqual(kwargs["params"]["output_format"], "mp3_22050_32")
        response.raise_for_status.assert_called_once_with()


class ConversationContractTests(unittest.TestCase):
    def test_typed_conversation_returns_timed_five_emotion_contract(self) -> None:
        names = PhraseTests.names
        result = TraceResult(
            response="I hear you.",
            response_ids=[],
            labels=[],
            names=names,
            scores=torch.zeros((1, len(names))),
            neutral_mean=torch.zeros(len(names)),
            neutral_std=torch.ones(len(names)),
            generation_seconds=0.2,
            replay_seconds=0.1,
        )
        scores = {name: 0.0 for name in SHARED_EMOTIONS}
        scores["happy"] = 1.0
        tagged = "[pleased] I hear you."
        text = "I hear you."
        start = tagged.index(text)
        phrases = [
            {
                "text": text,
                "emotion": "happy",
                "scores": scores,
                "intensity": 0.5,
                "evidence": 2.0,
                "shared_mass": 1.0,
                "direction": "[pleased]",
                "start_seconds": None,
                "end_seconds": None,
            }
        ]
        alignment = {
            "characters": list(tagged),
            "character_start_times_seconds": [
                index / 100 for index in range(len(tagged))
            ],
            "character_end_times_seconds": [
                (index + 1) / 100 for index in range(len(tagged))
            ],
        }
        speech = Mock()
        speech.synthesize_with_timestamps.return_value = (b"ID3audio", alignment)

        runtime = emotion_web.runtime
        saved = {
            name: getattr(runtime, name)
            for name in (
                "model",
                "tokenizer",
                "names",
                "vectors",
                "neutral_mean",
                "neutral_std",
                "speech",
                "jobs",
                "diagnostics",
            )
        }
        try:
            runtime.model = object()
            runtime.tokenizer = object()
            runtime.names = names
            runtime.vectors = torch.zeros((len(names), 2))
            runtime.neutral_mean = torch.zeros(len(names))
            runtime.neutral_std = torch.ones(len(names))
            runtime.speech = speech
            runtime.jobs = type(saved["jobs"])()
            runtime.diagnostics = type(saved["diagnostics"])()
            with (
                patch.object(
                    emotion_web,
                    "score_language_emotion",
                    return_value=(scores, 0.8, {"happy": {"raw": 0.2, "z": 1.0}}),
                ),
                patch.object(emotion_web, "analyze_prompt", return_value=result),
                patch.object(
                    emotion_web,
                    "shared_phrase_plan",
                    return_value=(tagged, phrases, [(start, start + len(text))]),
                ),
            ):
                silent_response = emotion_web.conversation(
                    ConversationRequest(
                        api_version=API_VERSION,
                        transcript="Today is looking up",
                        synthesize_speech=False,
                    )
                )
                silent_validated = ConversationResponse.model_validate(
                    silent_response
                )
                self.assertIsNone(silent_validated.speech_id)
                self.assertIsNone(silent_validated.timings.speech_seconds)
                speech.synthesize_with_timestamps.assert_not_called()

                response = emotion_web.conversation(
                    ConversationRequest(
                        api_version=API_VERSION,
                        transcript="Today is looking up",
                    )
                )

            validated = ConversationResponse.model_validate(response)
            self.assertEqual(validated.api_version, API_VERSION)
            self.assertIsNotNone(validated.speech_id)
            self.assertEqual(validated.human.dominant, "happy")
            self.assertEqual(
                validated.human.modalities["language"].confidence,
                0.35,
            )
            self.assertEqual(validated.response_context.effect, "words-only")
            self.assertEqual(validated.response_context.nonverbal_weight, 0)
            self.assertEqual(validated.phrases[0].emotion, "happy")
            self.assertIsNotNone(validated.phrases[0].start_seconds)
            self.assertIsNotNone(validated.phrases[0].end_seconds)
            self.assertEqual(
                runtime.jobs[validated.speech_id],
                b"ID3audio",
            )
            speech.synthesize_with_timestamps.assert_called_once_with(tagged)
        finally:
            for name, value in saved.items():
                setattr(runtime, name, value)


if __name__ == "__main__":
    unittest.main()
