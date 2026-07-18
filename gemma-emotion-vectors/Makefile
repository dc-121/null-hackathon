.PHONY: chat web

chat:
	uv run python scripts/emotion_trace.py --interactive --speak --timings --delivery-mode raw --tts-stability 0.25 --max-new-tokens 64

web:
	uv run python scripts/emotion_web.py
