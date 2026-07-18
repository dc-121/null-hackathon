# null-hackathon

> **Current integrated build: [NULL MIRROR](../README.md).** This document
> preserves the original human-simulation concept and frontend provenance. The
> root README is the authoritative runbook for the current Gemma, HSEmotion,
> browser-prosody, and ElevenLabs implementation, including the labeled
> same-transcript judge proof at `/?demo=1`.

**Theme: inter(action/faces).** Every "AI agents for x" release is a chat
interface with connectors, so nothing feels special — people never see the
magic. Show them the magic.

## The idea

Speech-to-text compresses speech down to words. Emphasis, hesitation, the crack
in your voice — all discarded before the model sees it. Even native
speech-to-speech models, which *do* hear tone, give you no interface to it. It
hears you and you have no idea what it heard.

We're building the complement of speech-to-text: the part it drops.

Two halves, both always visible:

1. **What it heard in you** — your prosody, live, as figures moving in a space.
2. **How it chose its words** — the words it almost said, ghosting behind the
   ones it picked.

The demo beat: run two generations, one that heard your emotion and one blind
to it, and show where they diverge. The diff *is* the empathy, made visible.

## Rules

- **Everything relative to your own baseline.** Never absolute labels. "anger
  0.82" is a claim about a stranger's inner life they can disprove on the spot;
  "40% above your own normal" can't be argued with.
- **The mute test.** Delete every word, number and label from the screen. Can
  someone still tell the moment you got agitated? If not, the graphics are
  decoration and the text is doing the work.
- **No "show thinking" button.** A button means the default state is a chat box
  with a drawer — exactly what the theme is mocking. Everything is always on.
- **Fast loop is browser-only.** Prosody, face, crowd motion: no network hops.
  Past ~100ms the causal coupling breaks and it reads as a laggy chart. Slow
  loop (utterance emotion, generation, transcription) can be a service.
- **Never put per-frame state in `useState`.** Mutate `src/state/emotion.ts`,
  read it inside your render loop. React owns the shell, not the animation.

## Run

```bash
pnpm install
pnpm dev
```

## Layout

```
src/
  state/emotion.ts   THE CONTRACT — shared shape, agree changes here first
  crowd/             the figures (canvas 2d)
  App.tsx            two panes: you | it
```

The divider is meant to be permeable — figures drift across, and the two crowds
sync when it attunes to you. Attunement made visible, no labels.

## Stack

| Layer | Pick |
|---|---|
| Shell | React + Vite + TS |
| Fast loop | `AnalyserNode` → RMS, pitch, spectral tilt @60fps |
| Slow loop | Hume EVI → Cowen emotion dimensions per utterance |
| Generation | OpenAI w/ `logprobs` + `top_logprobs` (EVI won't give us these) |
| Face in | MediaPipe Face Landmarker — on-device, 52 ARKit blendshapes |
| Biometrics | Garmin broadcast HR over Web Bluetooth (`0x180D`). Arousal only |
| Auth | Supabase |

## Build order

1. Voice → live figures. This alone is the project.
2. Word choice ghosting + the blind/heard diff.
3. MediaPipe face.
4. The avatar — **stylised, not photoreal**. A realistic face reacting slightly
   wrong is worse than an abstract one reacting perfectly.
5. Garmin last. BLE at a crowded venue is unreliable; record a fallback session.

## License

MIT — see [LICENSE](./LICENSE).
