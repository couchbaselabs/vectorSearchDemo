#!/usr/bin/env python3
"""Connectivity + readiness checks shared by setup.py (CLI) and app.py (web config drawer).

run_checks(settings) returns a list of {"name", "ok", "detail"} dicts so the caller can render
them however it likes (coloured CLI lines, or JSON for the UI). Every check is wrapped so one
failure doesn't abort the rest; a failed cluster connection short-circuits the checks that need it.

All heavy imports (couchbase, openai, sentence_transformers) are done lazily inside the checks so
importing this module stays cheap and a missing optional dependency surfaces as a failed check.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional

import config

# The collections this demo expects under the _default scope.
DEMO_COLLECTIONS = ["movies", "emails", "yelp"]


def _result(name: str, ok: bool, detail: str) -> Dict:
    return {"name": name, "ok": ok, "detail": detail}


# Bounded timeouts so the connectivity check always returns promptly instead of hanging the
# request (a slow/unreachable cluster or a bad OpenAI key would otherwise block for a long time,
# which the browser reports as "Failed to fetch").
CONNECT_TIMEOUT_SECONDS = 8
OPENAI_TIMEOUT_SECONDS = 10


def _connect(settings: "config.Settings", timeout_seconds: int = CONNECT_TIMEOUT_SECONDS):
    """Open a cluster connection and block until it's ready (bounded). Raises on failure.

    Sets explicit connect/bootstrap/KV timeouts — wait_until_ready alone doesn't bound the SDK's
    underlying operations, so a bad connection string or a blocked IP could otherwise hang far
    longer than timeout_seconds."""
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions, ClusterTimeoutOptions
    from couchbase.auth import PasswordAuthenticator

    t = timedelta(seconds=timeout_seconds)
    cluster = Cluster(
        settings.couchbase_connstr,
        ClusterOptions(
            PasswordAuthenticator(settings.couchbase_username, settings.couchbase_password),
            timeout_options=ClusterTimeoutOptions(
                connect_timeout=t,
                bootstrap_timeout=t,
                resolve_timeout=t,
                kv_timeout=t,
                management_timeout=t,
            ),
        ),
    )
    cluster.wait_until_ready(t)
    return cluster


def _check_embedding(settings: "config.Settings", results: List[Dict]) -> None:
    """Confirm the resolved provider can produce an embedding, and report its dimension."""
    provider = settings.provider
    model = settings.embedding_model
    try:
        if provider == "local":
            from sentence_transformers import SentenceTransformer

            vec = SentenceTransformer(model).encode("connectivity probe").tolist()
        else:
            from openai import OpenAI

            if not settings.openai_api_key:
                results.append(_result(
                    "Embedding provider", False,
                    "provider 'openai' selected but OPENAI_API_KEY is not set",
                ))
                return
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
            resp = client.embeddings.create(model=model, input="connectivity probe")
            vec = resp.data[0].embedding

        dim = len(vec)
        declared = settings.declared_dimensions
        if declared is not None and dim != declared:
            results.append(_result(
                "Embedding provider", False,
                f"provider '{provider}' model '{model}' produced {dim}-dim vectors but expected "
                f"{declared} — data, index, and query dimensions must agree",
            ))
        else:
            suffix = f" (matches expected {declared})" if declared is not None else ""
            results.append(_result(
                "Embedding provider", True,
                f"provider '{provider}' model '{model}' → {dim}-dim{suffix}",
            ))
    except Exception as e:  # noqa: BLE001 - surface any provider/model error as a failed check
        results.append(_result("Embedding provider", False, f"{provider} check failed: {e}"))


def _check_openai_for_rag(settings: "config.Settings", results: List[Dict]) -> None:
    """RAG generation always needs OpenAI (chat), even when embeddings are local."""
    if not settings.openai_api_key:
        results.append(_result(
            "OpenAI (for RAG)", False,
            "OPENAI_API_KEY is not set — loading works, but the rag*.py scripts need it for the LLM",
        ))
        return
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
        client.models.list()  # cheap call that validates the key/endpoint
        results.append(_result("OpenAI (for RAG)", True, f"key valid; chat model '{settings.openai_chat_model}'"))
    except Exception as e:  # noqa: BLE001
        results.append(_result("OpenAI (for RAG)", False, f"key/endpoint check failed: {e}"))


def run_checks(settings: "config.Settings", collections: Optional[List[str]] = None) -> List[Dict]:
    """Run the full readiness preflight. Returns a list of {name, ok, detail}."""
    collections = collections or DEMO_COLLECTIONS
    results: List[Dict] = []

    # 1. Couchbase connection. If this fails, skip the checks that need a live cluster.
    cluster = None
    try:
        cluster = _connect(settings)
        results.append(_result("Couchbase connection", True, f"reachable at {settings.couchbase_connstr}"))
    except Exception as e:  # noqa: BLE001
        results.append(_result("Couchbase connection", False, f"could not connect: {e}"))

    if cluster is not None:
        # 2. Bucket + 3. collections.
        try:
            cb_bucket = cluster.bucket(settings.couchbase_bucket)
            scopes = cb_bucket.collections().get_all_scopes()
            results.append(_result("Bucket", True, f"'{settings.couchbase_bucket}' found"))

            default = next((s for s in scopes if s.name == "_default"), None)
            existing = {c.name for c in default.collections} if default else set()
            present = [c for c in collections if c in existing]
            missing = [c for c in collections if c not in existing]
            if missing:
                results.append(_result(
                    "Collections", False,
                    f"present: {present or 'none'}; missing: {missing} "
                    f"(create them under _default — see README Step 0.5)",
                ))
            else:
                results.append(_result("Collections", True, f"found under _default: {present}"))
        except Exception as e:  # noqa: BLE001
            results.append(_result(
                "Bucket", False,
                f"could not open bucket '{settings.couchbase_bucket}': {e}",
            ))

    # 4. Embedding provider.
    _check_embedding(settings, results)
    # 5. OpenAI for RAG generation.
    _check_openai_for_rag(settings, results)

    return results


def all_ok(results: List[Dict]) -> bool:
    return all(r["ok"] for r in results)
