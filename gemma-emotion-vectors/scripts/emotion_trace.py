#!/usr/bin/env python3
"""Generate with Gemma 4 E4B and trace published emotion vectors per token."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-4-E4B-it"
VECTOR_REPO = "rain1955/emotion-vector-replication"
VECTOR_FILE = "results/emotion_vectors.npz"
TARGET_LAYER = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace Gemma's layer-28 activation against published emotion vectors."
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
    return model, tokenizer, names, vectors, perf_counter() - started


def analyze_prompt(
    prompt: str,
    *,
    model,
    tokenizer,
    names: list[str],
    vectors: torch.Tensor,
    max_new_tokens: int,
    top_k: int,
    timings: bool,
) -> None:
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

    response_ids = output_ids[0, prompt_length:]
    print("\n=== Gemma response ===")
    print(tokenizer.decode(response_ids, skip_special_tokens=True).strip())

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
        range(prompt_length, prompt_length + response_ids.numel())
    )
    labels = ["<before response>"] + [
        visible_token(tokenizer, int(token_id)) for token_id in response_ids
    ]
    normalized_hidden = torch.nn.functional.normalize(hidden[positions], dim=-1)
    scores = normalized_hidden @ vectors.T

    print("\n=== Emotion-vector trace (Gemma layer 28) ===")
    print("Cosine activations are directional scores, not probabilities.")
    top_k = min(top_k, len(names))
    for index, (label, row) in enumerate(zip(labels, scores, strict=True)):
        values, indices = row.topk(top_k)
        ranked = ", ".join(
            f"{names[int(emotion_index)]}={float(value):+.4f}"
            for value, emotion_index in zip(values, indices, strict=True)
        )
        print(f"[{index:02d}] {label!r}: {ranked}")
    if timings:
        print(
            f"\nTimings: generation {generation_seconds:.2f}s · "
            f"activation replay {replay_seconds:.2f}s"
        )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1 or args.top_k < 1:
        raise SystemExit("--max-new-tokens and --top-k must be positive")
    if not args.interactive and not args.prompt:
        raise SystemExit("Provide a prompt or use --interactive")

    model, tokenizer, names, vectors, load_seconds = load_runtime()
    if args.timings:
        print(f"Model load: {load_seconds:.2f}s")

    prompts = [args.prompt] if args.prompt else []
    for prompt in prompts:
        analyze_prompt(
            prompt,
            model=model,
            tokenizer=tokenizer,
            names=names,
            vectors=vectors,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            timings=args.timings,
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
            analyze_prompt(
                prompt,
                model=model,
                tokenizer=tokenizer,
                names=names,
                vectors=vectors,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                timings=args.timings,
            )


if __name__ == "__main__":
    main()
