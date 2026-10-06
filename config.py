#!/usr/bin/env python3
"""Central configuration for load.py and the rag*.py scripts (backlog #3).

Single source of truth for Couchbase + embedding settings. Reads the process environment
(populate it first with python-dotenv's load_dotenv()), validates per-provider requirements,
and resolves the effective embedding provider / model / dimension with a documented precedence:

    provider:   --embedding-provider  >  SENTENCE_TRANSFORMER_MODEL set → local, else openai
    model:      --embedding-model      >  SENTENCE_TRANSFORMER_MODEL / OPENAI_EMBEDDING_MODEL  >  built-in default
    dimensions: --dimensions           >  VECTOR_DIMENSIONS                                    >  derived from the model

Built on pydantic (already an installed dependency) rather than pydantic-settings so no new
runtime package is required.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# Output dimensions for known embedding models, used to derive the expected dimension when
# neither --dimensions nor VECTOR_DIMENSIONS is set. Keyed by bare name and "org/name" form.
KNOWN_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def known_dimension(model: str) -> Optional[int]:
    """Return the known output dimension for a model name, or None if unknown."""
    return KNOWN_DIMENSIONS.get(model) or KNOWN_DIMENSIONS.get(model.split("/")[-1])


def _as_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Settings(BaseModel):
    """Resolved configuration. Build with Settings.load(...); read the effective values via the
    provider / embedding_model / declared_dimensions properties."""

    # Couchbase
    couchbase_connstr: str = "couchbase://127.0.0.1"
    couchbase_username: str = "Administrator"
    couchbase_password: str = "password"
    couchbase_bucket: str = "vectorSearchDemo"  # used by the connectivity preflight / setup

    # Embedding provider inputs (from .env)
    sentence_transformer_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = DEFAULT_OPENAI_MODEL
    openai_chat_model: str = "gpt-4o-mini"
    vector_dimensions: Optional[int] = None
    hf_token: Optional[str] = None

    # Agent Memory (phase 2): base URL of the couchbase-agent-memory server the app talks to.
    # Defaults to a locally-run server on host port 8090 (app.py already uses 8080); repoint to
    # the Capella-hosted endpoint when available.
    agent_memory_base_url: str = "http://localhost:8090"
    # Global toggle: when true, RAG chats use the AI Data Plane (agent memory) features.
    ai_dataplane_enabled: bool = False

    # Per-invocation CLI overrides (highest precedence)
    provider_override: Optional[str] = None
    model_override: Optional[str] = None
    dimensions_override: Optional[int] = None

    @classmethod
    def load(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> "Settings":
        """Build Settings from the environment (after .env is loaded), applying CLI overrides."""
        return cls(
            couchbase_connstr=os.getenv("COUCHBASE_CONNSTR", "couchbase://127.0.0.1"),
            couchbase_username=os.getenv("COUCHBASE_USERNAME", "Administrator"),
            couchbase_password=os.getenv("COUCHBASE_PASSWORD", "password"),
            couchbase_bucket=os.getenv("COUCHBASE_BUCKET", "vectorSearchDemo"),
            sentence_transformer_model=os.getenv("SENTENCE_TRANSFORMER_MODEL") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            vector_dimensions=_as_int(os.getenv("VECTOR_DIMENSIONS")),
            hf_token=os.getenv("HF_TOKEN") or None,
            agent_memory_base_url=os.getenv("AGENT_MEMORY_BASE_URL", "http://localhost:8090"),
            ai_dataplane_enabled=(os.getenv("AI_DATAPLANE_ENABLED", "") or "").strip().lower()
            in ("1", "true", "yes", "on"),
            provider_override=provider,
            model_override=model,
            dimensions_override=dimensions,
        )

    @classmethod
    def from_env_dict(cls, env: "dict[str, str]") -> "Settings":
        """Build Settings from a plain dict of ENV-style keys (not the process environment).
        Used by the web config drawer to test values that haven't been written to .env yet."""
        return cls(
            couchbase_connstr=env.get("COUCHBASE_CONNSTR") or "couchbase://127.0.0.1",
            couchbase_username=env.get("COUCHBASE_USERNAME") or "Administrator",
            couchbase_password=env.get("COUCHBASE_PASSWORD") or "password",
            couchbase_bucket=env.get("COUCHBASE_BUCKET") or "vectorSearchDemo",
            sentence_transformer_model=env.get("SENTENCE_TRANSFORMER_MODEL") or None,
            openai_api_key=env.get("OPENAI_API_KEY") or None,
            openai_base_url=env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            openai_embedding_model=env.get("OPENAI_EMBEDDING_MODEL") or DEFAULT_OPENAI_MODEL,
            openai_chat_model=env.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
            vector_dimensions=_as_int(env.get("VECTOR_DIMENSIONS")),
            hf_token=env.get("HF_TOKEN") or None,
        )

    @property
    def provider(self) -> str:
        """Effective embedding provider: CLI override, else inferred from .env."""
        if self.provider_override:
            return self.provider_override
        return "local" if self.sentence_transformer_model else "openai"

    @property
    def embedding_model(self) -> str:
        """Effective embedding model name for the resolved provider."""
        if self.model_override:
            return self.model_override
        if self.provider == "local":
            return self.sentence_transformer_model or DEFAULT_LOCAL_MODEL
        return self.openai_embedding_model or DEFAULT_OPENAI_MODEL

    @property
    def declared_dimensions(self) -> Optional[int]:
        """Expected embedding dimension: --dimensions, else VECTOR_DIMENSIONS, else derived from
        the model. None means "unknown" (the runtime preflight then has nothing to compare against)."""
        if self.dimensions_override is not None:
            return self.dimensions_override
        if self.vector_dimensions is not None:
            return self.vector_dimensions
        return known_dimension(self.embedding_model)

    def validate_for_embedding(self) -> None:
        """Fail fast on missing per-provider requirements for computing embeddings."""
        if self.provider == "openai" and not self.openai_api_key:
            raise SystemExit(
                "OPENAI_API_KEY is required for the OpenAI embedding provider. "
                "Set it in .env, or use --embedding-provider local."
            )

    def validate_for_rag(self) -> None:
        """RAG scripts also need OpenAI for the chat completion, regardless of embedding provider."""
        self.validate_for_embedding()
        if not self.openai_api_key:
            raise SystemExit("OPENAI_API_KEY is required for RAG (LLM generation).")


# ---------------------------------------------------------------------------
# .env read/write helpers (used by setup.py and the web config drawer)
# ---------------------------------------------------------------------------

# Keys this project manages in .env, in a sensible file order.
ENV_KEYS = [
    "COUCHBASE_CONNSTR",
    "COUCHBASE_USERNAME",
    "COUCHBASE_PASSWORD",
    "COUCHBASE_BUCKET",
    "VECTOR_DIMENSIONS",
    "SENTENCE_TRANSFORMER_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_CHAT_MODEL",
    "OPENAI_BASE_URL",
    "HF_TOKEN",
    "AGENT_MEMORY_BASE_URL",
    "AI_DATAPLANE_ENABLED",
]

# Keys whose values should be masked when displayed back (e.g. in the web UI).
SECRET_KEYS = {"COUCHBASE_PASSWORD", "OPENAI_API_KEY", "HF_TOKEN"}


def read_env_file(path: str = ".env") -> "dict[str, str]":
    """Parse a .env file into a dict. Ignores comments/blank lines; strips surrounding quotes.
    Returns {} if the file doesn't exist."""
    values: "dict[str, str]" = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # Drop inline comments only when the value isn't quoted.
            val = val.strip()
            if val and val[0] in "\"'" and val[-1:] == val[0]:
                val = val[1:-1]
            else:
                val = val.split(" #", 1)[0].strip()
            values[key] = val
    return values


def write_env_file(updates: "dict[str, str]", path: str = ".env") -> None:
    """Merge `updates` into the .env file, preserving existing keys/values not being changed.
    A value of "" or None removes that key. Quotes values that contain spaces or '#'. Creates
    the file if missing."""
    current = read_env_file(path)
    for key, val in updates.items():
        if val is None or val == "":
            current.pop(key, None)
        else:
            current[key] = val

    ordered = [k for k in ENV_KEYS if k in current]
    extras = [k for k in current if k not in ENV_KEYS]
    lines = []
    for key in ordered + extras:
        val = current[key]
        if val and (" " in val or "#" in val):
            val = f'"{val}"'
        lines.append(f"{key}={val}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def mask_secret(value: "Optional[str]") -> str:
    """Mask a secret for display: keep the last 4 chars, hide the rest. Empty stays empty."""
    if not value:
        return ""
    if len(value) <= 4:
        return "•" * len(value)
    return "•" * (len(value) - 4) + value[-4:]


# ---------------------------------------------------------------------------
# Web config-drawer field mapping (fields <-> .env keys)
# ---------------------------------------------------------------------------

def form_from_env(env: "dict[str, str]") -> "dict[str, str]":
    """Build the drawer's display payload from .env values, with secrets masked."""
    provider = "local" if env.get("SENTENCE_TRANSFORMER_MODEL") else (
        "openai" if env.get("OPENAI_API_KEY") else "local")
    model = env.get("SENTENCE_TRANSFORMER_MODEL") if provider == "local" else env.get("OPENAI_EMBEDDING_MODEL", "")
    return {
        "connstr": env.get("COUCHBASE_CONNSTR", ""),
        "username": env.get("COUCHBASE_USERNAME", ""),
        "password": mask_secret(env.get("COUCHBASE_PASSWORD", "")),
        "bucket": env.get("COUCHBASE_BUCKET", "vectorSearchDemo"),
        "provider": provider,
        "model": model or "",
        "dimensions": env.get("VECTOR_DIMENSIONS", ""),
        "openai_key": mask_secret(env.get("OPENAI_API_KEY", "")),
        "openai_chat_model": env.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    }


def env_updates_from_form(form: "dict[str, str]", existing: "Optional[dict]" = None) -> "dict[str, str]":
    """Map drawer form fields back to .env key/values. A secret field left masked (contains the
    mask char) or omitted keeps the existing stored value rather than overwriting it."""
    existing = existing if existing is not None else read_env_file()

    def keep(new: "Optional[str]", key: str) -> str:
        if new is None or "•" in new:
            return existing.get(key, "")
        return new

    provider = (form.get("provider") or "local").lower()
    model = (form.get("model") or "").strip()
    env = {
        "COUCHBASE_CONNSTR": (form.get("connstr") or "").strip(),
        "COUCHBASE_USERNAME": (form.get("username") or "").strip(),
        "COUCHBASE_PASSWORD": keep(form.get("password"), "COUCHBASE_PASSWORD"),
        "COUCHBASE_BUCKET": (form.get("bucket") or "vectorSearchDemo").strip(),
        "VECTOR_DIMENSIONS": (form.get("dimensions") or "").strip(),
        "OPENAI_API_KEY": keep(form.get("openai_key"), "OPENAI_API_KEY"),
        "OPENAI_CHAT_MODEL": (form.get("openai_chat_model") or "gpt-4o-mini").strip(),
    }
    if provider == "local":
        env["SENTENCE_TRANSFORMER_MODEL"] = model or "all-MiniLM-L6-v2"
    else:
        # Clearing the local model is what makes the provider resolve to openai.
        env["SENTENCE_TRANSFORMER_MODEL"] = ""
        env["OPENAI_EMBEDDING_MODEL"] = model or "text-embedding-3-small"
    return env
