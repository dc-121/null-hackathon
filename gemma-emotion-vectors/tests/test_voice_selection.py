import unittest

import torch

from scripts.emotion_trace import (
    EmotionComponent,
    compile_direction,
    select_voice_emotion,
)


class SelectVoiceEmotionTests(unittest.TestCase):
    names = ["angry", "happy", "sad"]

    def test_weak_evidence_stays_neutral(self) -> None:
        selected = select_voice_emotion(
            torch.tensor([1.24, 0.80, 0.20]), self.names, None
        )
        self.assertIsNone(selected)

    def test_strongest_evidence_wins_without_previous_direction(self) -> None:
        selected = select_voice_emotion(
            torch.tensor([1.40, 3.20, 0.30]), self.names, None
        )
        self.assertEqual(selected, 1)

    def test_small_lead_does_not_switch_direction(self) -> None:
        selected = select_voice_emotion(
            torch.tensor([1.50, 1.90, 0.30]), self.names, "angry"
        )
        self.assertEqual(selected, 0)

    def test_large_lead_switches_direction(self) -> None:
        selected = select_voice_emotion(
            torch.tensor([5.00, 6.10, 0.30]), self.names, "angry"
        )
        self.assertEqual(selected, 1)

    def test_faded_previous_direction_does_not_block_switch(self) -> None:
        selected = select_voice_emotion(
            torch.tensor([1.00, 1.40, 0.30]), self.names, "angry"
        )
        self.assertEqual(selected, 1)

    def test_extreme_direction_uses_one_emotion_and_matching_performance_cue(self) -> None:
        component = EmotionComponent(
            name="happy",
            weight=0.60,
            evidence=7.00,
            raw=0.10,
            delta=0.05,
            z=5.00,
            delta_z=4.00,
            intensity=1.00,
        )
        self.assertEqual(
            compile_direction(component, delivery_mode="raw"),
            "[ecstatic] [laughing]",
        )

    def test_neutral_direction_emits_no_tag(self) -> None:
        self.assertEqual(compile_direction(None, delivery_mode="raw"), "")


if __name__ == "__main__":
    unittest.main()
