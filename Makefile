SHELL := /bin/sh

BACKEND_DIR := gemma-emotion-vectors
FRONTEND_DIR := experience
PNPM_STORE := $(CURDIR)/.pnpm-store
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

.PHONY: setup build run backend frontend test

setup:
	cd $(BACKEND_DIR) && uv sync --python 3.12
	pnpm --dir $(FRONTEND_DIR) install --store-dir $(PNPM_STORE)

build:
	pnpm --dir $(FRONTEND_DIR) build

run: build
	cd $(BACKEND_DIR) && uv run python scripts/emotion_web.py

backend:
	cd $(BACKEND_DIR) && uv run python scripts/emotion_web.py --no-open

frontend:
	pnpm --dir $(FRONTEND_DIR) dev

test:
	cd $(BACKEND_DIR) && uv run python -m unittest discover -s tests -v
	pnpm --dir $(FRONTEND_DIR) build
