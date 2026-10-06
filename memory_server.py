#!/usr/bin/env python3
"""Native Agent Memory server for the demo — a side-by-side process (no Docker) that the real
`couchbase-agent-memory` SDK (AgentMemoryClient) talks to, backed by your Couchbase cluster.

It implements the subset of the Agent Memory HTTP API the SDK calls — users, sessions, and memory
(add / search / list / delete) plus health — and matches the SDK's response shapes so the real SDK
works unchanged. Memories are stored in the `agentmemory` scope (collections: users / sessions /
memory) and recalled by vector similarity on OpenAI embeddings.

Honest simplification vs. the official server: `add_memory` here is SYNCHRONOUS (embed + store) and
we do NOT run the official server's background LLM "fact extraction" — stored blocks keep the raw
message and recall by semantic similarity. Point `AGENT_MEMORY_BASE_URL` at the real/Capella server
later and the same SDK calls hit it instead; no app changes.

Run:  python memory_server.py        # serves on http://localhost:8090
Config: AGENTMEMORY_* env (falls back to COUCHBASE_* / OPENAI_API_KEY from .env).
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

SCOPE = "agentmemory"
COLL_USERS = "users"
COLL_SESSIONS = "sessions"
COLL_MEMORY = "memory"
DEFAULT_RELEVANT_K = 5
# Minimum cosine similarity for a memory to be recalled. Repeats (~0.9+) and related questions
# (~0.3-0.6) pass; unrelated questions (~0.05) are filtered out so they don't inject noise.
# Override per deployment with AGENTMEMORY_MIN_SCORE, or per request via filters.min_score.
DEFAULT_MIN_SCORE = float(os.getenv("AGENTMEMORY_MIN_SCORE", "0.2"))

app = Flask(__name__)
_STARTED = time.time()

# Live, in-process usage counters so the demo's status overlay can show real activity
# (memories written, searches run, context served from memory). Reset when the server restarts.
STATS = {
    "memory_added": 0,          # memory blocks stored
    "searches": 0,              # search_memory calls
    "blocks_recalled": 0,       # blocks returned by searches
    "chars_served_from_memory": 0,  # characters of recalled content (for a tokens-saved estimate)
}


def _est_tokens(chars: int) -> int:
    """Rough token estimate (~4 chars/token). Labeled as an estimate in the UI."""
    return round(chars / 4)


def _env(name: str, *fallbacks: str, default: str = "") -> str:
    for key in (name, *fallbacks):
        val = os.getenv(key)
        if val:
            return val
    return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Couchbase (KV only — no query index required) + OpenAI embeddings
# ---------------------------------------------------------------------------

_scope = None
_openai = None


def _get_scope():
    """Connect once and return the `agentmemory` scope. Bounded timeouts so a bad config fails fast."""
    global _scope
    if _scope is None:
        from datetime import timedelta
        from couchbase.cluster import Cluster
        from couchbase.options import ClusterOptions, ClusterTimeoutOptions
        from couchbase.auth import PasswordAuthenticator

        connstr = _env("AGENTMEMORY_CONN_STRING", "COUCHBASE_CONNSTR", default="couchbase://127.0.0.1")
        username = _env("AGENTMEMORY_USERNAME", "COUCHBASE_USERNAME", default="Administrator")
        password = _env("AGENTMEMORY_PASSWORD", "COUCHBASE_PASSWORD", default="password")
        bucket = _env("AGENTMEMORY_BUCKET", "COUCHBASE_BUCKET", default="vectorSearchDemo")

        t = timedelta(seconds=10)
        cluster = Cluster(
            connstr,
            ClusterOptions(
                PasswordAuthenticator(username, password),
                timeout_options=ClusterTimeoutOptions(connect_timeout=t, bootstrap_timeout=t, kv_timeout=t),
            ),
        )
        cluster.wait_until_ready(t)
        _scope = cluster.bucket(bucket).scope(SCOPE)
    return _scope


def _coll(name: str):
    return _get_scope().collection(name)


def _embed(text: str) -> List[float]:
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(
            api_key=_env("OPENAI_API_KEY"),
            base_url=_env("OPENAI_BASE_URL", default="https://api.openai.com/v1"),
            timeout=20,
        )
    model = _env("AGENTMEMORY_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")
    return _openai.embeddings.create(model=model, input=text).data[0].embedding


def _cosine(a: List[float], b: List[float]) -> float:
    import numpy as np
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def _skey(user_id: str, session_id: str) -> str:
    return f"{user_id}::{session_id}"


def _get(coll, key):
    from couchbase.exceptions import DocumentNotFoundException
    try:
        return coll.get(key).content_as[dict]
    except DocumentNotFoundException:
        return None


# ---------------------------------------------------------------------------
# Response shaping (strip internal fields the SDK models don't expect)
# ---------------------------------------------------------------------------

def _block_public(doc: dict, rel_score: Optional[float] = None) -> dict:
    return {
        "block_id": doc["block_id"],
        "user_id": doc["user_id"],
        "session_id": doc["session_id"],
        "message": doc.get("message"),
        "fact": doc.get("fact"),
        "ingested_at": doc.get("ingested_at"),
        "created_at": doc.get("created_at"),
        "last_queued_at": None,
        "fail_count": 0,
        "annotations": doc.get("annotations"),
        "summary": doc.get("summary"),
        "contexts": doc.get("contexts"),
        "rel_score": rel_score,
    }


def _session_public(doc: dict) -> dict:
    return {
        "user_id": doc["user_id"],
        "session_id": doc["session_id"],
        "start_time": doc["start_time"],
        "end_time": doc.get("end_time"),
        "annotations": doc.get("annotations"),
        "metadata": doc.get("metadata"),
        "blocks_ttl": doc.get("blocks_ttl"),
    }


# ---------------------------------------------------------------------------
# Health (the SDK's health_ping checks several /health/* entities)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "healthy", "version": "0.1.0-demo",
                    "uptime_seconds": round(time.time() - _STARTED, 1)})


@app.get("/health/couchbase")
def health_couchbase():
    try:
        _get_scope()
        return jsonify({"status": "healthy"})
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "unhealthy", "detail": str(e)}), 503


@app.get("/health/models")
def health_models():
    model = _env("AGENTMEMORY_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")
    llm = _env("AGENTMEMORY_LLM_MODEL", "OPENAI_CHAT_MODEL", default="gpt-4o-mini")
    return jsonify({"embedding": {"status": "healthy", "model": model},
                    "llm": {"status": "healthy", "model": llm}})


@app.get("/health/async-batch-processor")
@app.get("/health/async-batch-processor-stats")
@app.get("/health/memory")
def health_optional():
    # We don't run the async batch processor (synchronous add_memory) — "not_initialized" is
    # treated as OK by the SDK for optional components.
    return jsonify({"status": "not_initialized"})


@app.get("/stats")
def stats():
    """Live usage counters for the demo's status overlay (not part of the SDK API)."""
    return jsonify({
        **STATS,
        "tokens_served_from_memory_est": _est_tokens(STATS["chars_served_from_memory"]),
        "uptime_seconds": round(time.time() - _STARTED, 1),
    })


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def _user_public(doc: dict) -> dict:
    return {"id": doc["id"], "name": doc.get("name"),
            "sessions": doc.get("sessions", []), "metadata": doc.get("metadata")}


@app.post("/users")
def create_user():
    body = request.get_json(silent=True) or {}
    user_id = body["user_id"]
    if _get(_coll(COLL_USERS), user_id):
        return jsonify({"error": "user already exists"}), 409
    doc = {"id": user_id, "name": body.get("name", user_id),
           "sessions": [], "metadata": body.get("metadata")}
    _coll(COLL_USERS).upsert(user_id, doc)
    return jsonify(_user_public(doc))


@app.post("/users/search")
def search_user():
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id")
    doc = _get(_coll(COLL_USERS), user_id)
    if not doc:
        return jsonify({"error": "not found"}), 404
    return jsonify(_user_public(doc))


@app.get("/users")
def list_users():
    # Listing all users would require a query index; not needed by the demo flow.
    return jsonify({"users": [], "count": 0})


@app.put("/users/<user_id>")
def put_user(user_id):
    body = request.get_json(silent=True) or {}
    doc = _get(_coll(COLL_USERS), user_id) or {"id": user_id, "sessions": [], "metadata": None}
    if "name" in body:
        doc["name"] = body["name"]
    doc.setdefault("name", user_id)
    if "metadata" in body:
        doc["metadata"] = body["metadata"]
    _coll(COLL_USERS).upsert(user_id, doc)
    return jsonify(_user_public(doc))


@app.delete("/users/<user_id>")
def delete_user(user_id):
    from couchbase.exceptions import DocumentNotFoundException
    user = _get(_coll(COLL_USERS), user_id)
    for sid in (user or {}).get("sessions", []):
        sess = _get(_coll(COLL_SESSIONS), _skey(user_id, sid))
        for bid in (sess or {}).get("blocks", []):
            try:
                _coll(COLL_MEMORY).remove(bid)
            except DocumentNotFoundException:
                pass
        try:
            _coll(COLL_SESSIONS).remove(_skey(user_id, sid))
        except DocumentNotFoundException:
            pass
    try:
        _coll(COLL_USERS).remove(user_id)
    except DocumentNotFoundException:
        pass
    return jsonify({"message": f"deleted user {user_id}"})


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/users/<user_id>/sessions")
def create_session(user_id):
    body = request.get_json(silent=True) or {}
    session_id = body["session_id"]
    key = _skey(user_id, session_id)
    if _get(_coll(COLL_SESSIONS), key):
        return jsonify({"error": "session already exists"}), 409
    doc = {
        "user_id": user_id, "session_id": session_id, "start_time": _now(),
        "end_time": None, "annotations": body.get("annotations"),
        "metadata": body.get("metadata"), "blocks_ttl": body.get("memory_blocks_ttl"),
        "blocks": [],
    }
    _coll(COLL_SESSIONS).upsert(key, doc)
    # register the session on the user doc
    user = _get(_coll(COLL_USERS), user_id) or {"id": user_id, "name": user_id, "sessions": [], "metadata": None}
    if session_id not in user.get("sessions", []):
        user.setdefault("sessions", []).append(session_id)
        _coll(COLL_USERS).upsert(user_id, user)
    return jsonify(_session_public(doc))


@app.get("/users/<user_id>/sessions/<session_id>")
def get_session(user_id, session_id):
    doc = _get(_coll(COLL_SESSIONS), _skey(user_id, session_id))
    if not doc:
        return jsonify({"error": "not found"}), 404
    return jsonify(_session_public(doc))


@app.get("/users/<user_id>/sessions")
def list_sessions(user_id):
    user = _get(_coll(COLL_USERS), user_id) or {"sessions": []}
    out = []
    for sid in user.get("sessions", []):
        doc = _get(_coll(COLL_SESSIONS), _skey(user_id, sid))
        if doc:
            out.append(_session_public(doc))
    return jsonify({"sessions": out, "count": len(out)})


@app.put("/users/<user_id>/sessions/<session_id>")
def update_session(user_id, session_id):
    key = _skey(user_id, session_id)
    doc = _get(_coll(COLL_SESSIONS), key)
    if not doc:
        return jsonify({"error": "not found"}), 404
    body = request.get_json(silent=True) or {}
    for f in ("annotations", "metadata"):
        if f in body:
            doc[f] = body[f]
    _coll(COLL_SESSIONS).upsert(key, doc)
    return jsonify(_session_public(doc))


@app.post("/users/<user_id>/sessions/<session_id>/end")
def end_session(user_id, session_id):
    key = _skey(user_id, session_id)
    doc = _get(_coll(COLL_SESSIONS), key)
    if not doc:
        return jsonify({"error": "not found"}), 404
    doc["end_time"] = _now()
    _coll(COLL_SESSIONS).upsert(key, doc)
    return jsonify(_session_public(doc))


@app.delete("/users/<user_id>/sessions/<session_id>")
def delete_session(user_id, session_id):
    from couchbase.exceptions import DocumentNotFoundException
    key = _skey(user_id, session_id)
    doc = _get(_coll(COLL_SESSIONS), key)
    for bid in (doc or {}).get("blocks", []):
        try:
            _coll(COLL_MEMORY).remove(bid)
        except DocumentNotFoundException:
            pass
    try:
        _coll(COLL_SESSIONS).remove(key)
    except DocumentNotFoundException:
        pass
    return jsonify({"message": f"deleted session {session_id}"})


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _message_text(msg: dict) -> str:
    return " ".join(x for x in (msg.get("user_content", ""), msg.get("assistant_content", "")) if x).strip()


@app.post("/users/<user_id>/sessions/<session_id>/memory")
def add_memory(user_id, session_id):
    skey = _skey(user_id, session_id)
    sess = _get(_coll(COLL_SESSIONS), skey)
    if not sess:
        return jsonify({"error": "session not found"}), 404
    body = request.get_json(silent=True) or {}
    messages = body.get("messages") or []
    facts = body.get("facts") or []
    annotations = body.get("annotations")

    block_ids: List[str] = []
    for item in messages:
        # Embed on the USER question so a repeat/related question matches closely (the stored answer
        # is still kept for the short-circuit return). Falls back to the full turn if no user text.
        embed_text = (item.get("user_content") or "").strip() or _message_text(item)
        block_ids.append(_store_block(user_id, session_id, message=item, fact=None,
                                      text=embed_text, annotations=annotations))
    for fact in facts:
        block_ids.append(_store_block(user_id, session_id, message=None, fact=fact,
                                      text=fact, annotations=annotations))

    sess.setdefault("blocks", []).extend(block_ids)
    _coll(COLL_SESSIONS).upsert(skey, sess)
    STATS["memory_added"] += len(block_ids)
    return jsonify({"message": f"added {len(block_ids)} block(s)", "accepted_count": len(block_ids),
                    "block_ids": block_ids, "rejected_count": 0, "rejected_details": None})


def _store_block(user_id, session_id, message, fact, text, annotations) -> str:
    block_id = uuid.uuid4().hex
    doc = {
        "block_id": block_id, "user_id": user_id, "session_id": session_id,
        "message": message, "fact": fact, "ingested_at": _now(), "created_at": _now(),
        "annotations": annotations, "summary": None, "contexts": None,
        "embedding": _embed(text) if text else None,
    }
    _coll(COLL_MEMORY).upsert(block_id, doc)
    return block_id


def _collect_blocks(user_id: str, session_id: str, session_ids) -> List[dict]:
    """Gather candidate blocks for a user, scoped by session_ids (None=current, 'all', or a list)."""
    user = _get(_coll(COLL_USERS), user_id) or {"sessions": []}
    if session_ids == "all":
        sids = user.get("sessions", [])
    elif isinstance(session_ids, list) and session_ids:
        sids = session_ids
    else:
        sids = [session_id]
    blocks = []
    for sid in sids:
        sess = _get(_coll(COLL_SESSIONS), _skey(user_id, sid))
        for bid in (sess or {}).get("blocks", []):
            doc = _get(_coll(COLL_MEMORY), bid)
            if doc:
                blocks.append(doc)
    return blocks


@app.post("/users/<user_id>/sessions/<session_id>/memory/search")
def search_memory(user_id, session_id):
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    filters = body.get("filters") or {}
    session_ids = filters.get("session_ids")
    relevant_k = filters.get("relevant_k") or DEFAULT_RELEVANT_K
    min_score = filters.get("min_score")
    if min_score is None:
        min_score = DEFAULT_MIN_SCORE

    blocks = _collect_blocks(user_id, session_id, session_ids)

    if query:
        qvec = _embed(query)
        scored = [(b, _cosine(qvec, b["embedding"])) for b in blocks if b.get("embedding")]
        scored.sort(key=lambda x: x[1], reverse=True)
        # Keep only sufficiently-relevant blocks (repeats + related), capped at relevant_k.
        top = [(b, s) for b, s in scored if s >= min_score][:relevant_k]
        result = [_block_public(b, rel_score=round(s, 4)) for b, s in top]
    else:
        blocks.sort(key=lambda b: b.get("ingested_at", ""), reverse=True)
        result = [_block_public(b) for b in blocks[:relevant_k]]

    STATS["searches"] += 1
    STATS["blocks_recalled"] += len(result)
    STATS["chars_served_from_memory"] += sum(
        len(_message_text(b["message"]) if b.get("message") else (b.get("fact") or "")) for b in result
    )
    return jsonify({"memory_blocks": result, "count": len(result)})


@app.get("/users/<user_id>/memory")
def list_memories(user_id):
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    session_ids_arg = request.args.get("session_ids", "")
    session_ids = "all" if session_ids_arg == "all" else (session_ids_arg.split(",") if session_ids_arg else None)
    blocks = _collect_blocks(user_id, session_ids[0] if isinstance(session_ids, list) else "", session_ids)
    blocks.sort(key=lambda b: b.get("ingested_at", ""), reverse=True)
    page = blocks[offset:offset + limit]
    return jsonify({"memory_blocks": [_block_public(b) for b in page],
                    "count": len(page), "total": len(blocks), "limit": limit, "offset": offset})


@app.delete("/users/<user_id>/sessions/<session_id>/memory")
def delete_memory(user_id, session_id):
    from couchbase.exceptions import DocumentNotFoundException
    body = request.get_json(silent=True) or {}
    requested = body.get("block_ids")
    skey = _skey(user_id, session_id)
    sess = _get(_coll(COLL_SESSIONS), skey) or {"blocks": []}
    existing = sess.get("blocks", [])
    targets = existing if requested == "all" else [b for b in (requested or []) if b in existing]
    deleted = 0
    for bid in targets:
        try:
            _coll(COLL_MEMORY).remove(bid)
            deleted += 1
        except DocumentNotFoundException:
            pass
    sess["blocks"] = [b for b in existing if b not in targets]
    _coll(COLL_SESSIONS).upsert(skey, sess)
    return jsonify({"deleted_count": deleted})


if __name__ == "__main__":
    port = int(os.getenv("AGENT_MEMORY_PORT", "8090"))
    print(f"Agent Memory (demo) server on http://localhost:{port}  (scope '{SCOPE}')")
    app.run(host="127.0.0.1", port=port, threaded=True)
