#!/usr/bin/env python3
"""Interactive configuration wizard — write .env without editing it by hand.

Run:  python setup.py

Prompts for the Couchbase connection details (from the Capella **Connect** tab; see README
Step 0.5), the embedding provider, and the OpenAI key if needed. Existing .env values are shown
as defaults so re-running only updates what you change. At the end it can run the connectivity
preflight so you know everything works before loading any data.
"""

from __future__ import annotations

import sys

import config
import preflight

MASK_SENTINEL = "\0keep\0"  # internal marker: user pressed Enter to keep an existing secret


def ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt with an optional default. For secrets, show a masked default and keep it on Enter."""
    if secret and default:
        shown = config.mask_secret(default)
        raw = input(f"{prompt} [{shown}] (Enter to keep): ").strip()
        return default if raw == "" else raw
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return default if raw == "" else raw
    return input(f"{prompt}: ").strip()


def ask_choice(prompt: str, choices: list, default: str) -> str:
    opts = "/".join(choices)
    while True:
        raw = input(f"{prompt} ({opts}) [{default}]: ").strip().lower()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"  Please enter one of: {opts}")


def main() -> None:
    print("\n=== vectorSearchDemo configuration ===\n")
    print("This writes .env for you. Press Enter to accept the [default] shown.\n")

    existing = config.read_env_file()
    updates = {}

    # --- Couchbase (from the Capella Connect tab) ---
    print("-- Couchbase Capella connection (see README Step 0.5) --")
    updates["COUCHBASE_CONNSTR"] = ask(
        "Connection string (Public Connection String from the Connect tab)",
        existing.get("COUCHBASE_CONNSTR", ""),
    )
    updates["COUCHBASE_USERNAME"] = ask("Database username", existing.get("COUCHBASE_USERNAME", ""))
    updates["COUCHBASE_PASSWORD"] = ask("Database password", existing.get("COUCHBASE_PASSWORD", ""), secret=True)
    updates["COUCHBASE_BUCKET"] = ask("Bucket name", existing.get("COUCHBASE_BUCKET", "vectorSearchDemo"))

    # --- Embedding provider ---
    print("\n-- Embedding provider --")
    default_provider = "local" if existing.get("SENTENCE_TRANSFORMER_MODEL") else (
        "openai" if existing.get("OPENAI_API_KEY") else "local")
    provider = ask_choice("Provider", ["local", "openai"], default_provider)

    if provider == "local":
        model = ask("Local model name", existing.get("SENTENCE_TRANSFORMER_MODEL") or "all-MiniLM-L6-v2")
        updates["SENTENCE_TRANSFORMER_MODEL"] = model
        dim = config.known_dimension(model)
        updates["VECTOR_DIMENSIONS"] = str(dim) if dim else ask(
            "Vector dimensions", existing.get("VECTOR_DIMENSIONS", "384"))
    else:
        # OpenAI: clear the local model so the provider resolves to openai.
        updates["SENTENCE_TRANSFORMER_MODEL"] = ""
        model = ask("OpenAI embedding model", existing.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small")
        updates["OPENAI_EMBEDDING_MODEL"] = model
        dim = config.known_dimension(model)
        updates["VECTOR_DIMENSIONS"] = str(dim) if dim else ask(
            "Vector dimensions", existing.get("VECTOR_DIMENSIONS", "1536"))

    # --- OpenAI (always needed for RAG generation) ---
    print("\n-- OpenAI (required for the RAG scripts; embeddings too if provider is openai) --")
    updates["OPENAI_API_KEY"] = ask("OpenAI API key", existing.get("OPENAI_API_KEY", ""), secret=True)
    chat = ask("OpenAI chat model", existing.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini")
    updates["OPENAI_CHAT_MODEL"] = chat

    config.write_env_file(updates)
    print("\n✓ Wrote .env\n")

    # --- Optional connectivity test ---
    if ask_choice("Run the connectivity test now?", ["y", "n"], "y") == "y":
        run_preflight()


def run_preflight() -> None:
    """Load the freshly-written .env and run the readiness checks, printing coloured lines."""
    from dotenv import load_dotenv

    load_dotenv(override=True)
    settings = config.Settings.load()

    print("\n=== Connectivity / readiness check ===\n")
    results = preflight.run_checks(settings)
    for r in results:
        mark = "\033[32m✓\033[0m" if r["ok"] else "\033[31m✗\033[0m"
        print(f"  {mark} {r['name']}: {r['detail']}")

    if preflight.all_ok(results):
        print("\n\033[32mAll checks passed — you're ready to load data.\033[0m\n")
    else:
        print("\n\033[31mSome checks failed. Fix the items above, then re-run: python setup.py\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(1)
