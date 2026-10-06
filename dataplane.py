#!/usr/bin/env python3
"""AI Data Plane integration helpers (phase 2).

Thin layer over the real `couchbase-agent-memory` SDK + the native memory server, used by:
- the RAG scripts, to recall prior context and remember new turns (when the global toggle is on);
- app.py, to report live status/usage for the diagram overlay and to flip the toggle.

Everything is best-effort: if the memory server is down or the SDK isn't installed, the helpers
degrade gracefully so the plain RAG path always still works.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import config

DEFAULT_RECALL_K = 5
DEFAULT_USER = "demo-user"  # single demo user; sessions separate conversations
# A recalled memory at/above this cosine is treated as "the same question" — we answer from it and
# skip the vector search + LLM entirely (the measurable "faster on repeats" win). Configurable.
DEFAULT_HIT_THRESHOLD = 0.9


def est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return round(len(text or "") / 4)


def print_timings(timings: "dict", extra: str = "") -> None:
    """Print a per-stage timing breakdown line for the CLI demo."""
    if not timings:
        return
    parts = " · ".join(f"{k} {v:.2f}s" for k, v in timings.items())
    total = sum(timings.values())
    line = f"\nTiming: {parts} · total {total:.2f}s"
    if extra:
        line += f"\n{extra}"
    print(line)


def augment_context(settings: "config.Settings", session_id: str, prompt: str, context: str,
                    min_score: Optional[float] = None):
    """If the AI Data Plane is enabled, prepend recalled memory to the RAG context and return
    (new_context, n_recalled). A no-op (returns the original context, 0) when disabled or on error.
    min_score gates recall by relevance (None = server default)."""
    if not settings.ai_dataplane_enabled:
        return context, 0
    mem, n = recall(settings, DEFAULT_USER, session_id, prompt, min_score=min_score)
    if mem:
        context = f"Relevant memory from earlier:\n{mem}\n\n---\n\n{context}"
    return context, n


# Phrases that indicate the RAG step couldn't answer. We must NOT cache these: a short-circuit on a
# later repeat would serve the stale non-answer instead of letting RAG try again.
_LOW_VALUE_MARKERS = (
    # "no info in the context" style (phrasing varies a lot across LLM runs)
    "not contain", "no information", "no relevant", "no matching context",
    "no usable context", "cannot find", "could not find", "couldn't find",
    "don't have", "do not have", "doesn't have", "insufficient", "not enough information",
    "not mention", "no mention", "not provide", "not specify", "not reference",
    "not include any", "unable to find", "there is no", "there are no",
    "no specific", "i'm sorry", "i am sorry", "i don't know", "i do not know",
)


def is_low_value_answer(answer: str) -> bool:
    """True if the answer looks like a RAG failure (empty or 'no info' style)."""
    a = (answer or "").strip().lower()
    if not a:
        return True
    return any(m in a for m in _LOW_VALUE_MARKERS)


def maybe_remember(settings: "config.Settings", session_id: str, prompt: str, answer: str) -> None:
    """Store this Q&A turn as memory when the AI Data Plane is enabled (best-effort). Skips
    low-value / failed answers so they don't poison later short-circuits."""
    if not settings.ai_dataplane_enabled:
        return
    if is_low_value_answer(answer):
        LOG = __import__("logging").getLogger("rag")
        LOG.info("AI Data Plane: not caching a low-value answer for session '%s'", session_id)
        return
    remember(settings, DEFAULT_USER, session_id, prompt, answer)


def _client(settings: "config.Settings"):
    from agentmemory import AgentMemoryClient
    return AgentMemoryClient(base_url=settings.agent_memory_base_url)


def ensure_session(client, user_id: str, session_id: str) -> None:
    """Get-or-create the user and session so add/search have somewhere to write."""
    from agentmemory.exceptions import NotFoundError, ConflictError
    try:
        user = client.get_user(user_id)
    except NotFoundError:
        user = client.create_user(user_id, name=user_id)
    try:
        user.create_session(session_id)
    except ConflictError:
        pass


def recall_blocks(settings: "config.Settings", user_id: str, session_id: str, query: str,
                  k: int = DEFAULT_RECALL_K, min_score: Optional[float] = None) -> List[dict]:
    """Return relevant memory blocks as [{text, user_content, assistant_content, score}] (ranked).
    Empty list on any error. min_score gates by cosine relevance (None = memory server default)."""
    try:
        with _client(settings) as client:
            ensure_session(client, user_id, session_id)
            sess = client.get_user(user_id).get_session(session_id)
            filters = {"relevant_k": k}
            if min_score is not None:
                filters["min_score"] = min_score
            res = sess.search_memory(query=query, filters=filters)
            out = []
            for b in res.memory_blocks:
                uc = b.message.user_content if b.message else ""
                ac = b.message.assistant_content if b.message else ""
                text = (" ".join(x for x in (uc, ac) if x)) if b.message else (b.fact or "")
                if text.strip():
                    out.append({"text": text.strip(), "user_content": uc or "",
                                "assistant_content": ac or "", "score": b.rel_score or 0.0})
            return out
    except Exception:  # noqa: BLE001 - memory is best-effort; never break RAG
        return []


def recall(settings: "config.Settings", user_id: str, session_id: str, query: str,
           k: int = DEFAULT_RECALL_K, min_score: Optional[float] = None) -> Tuple[str, int]:
    """Return (context_text, n_blocks) of relevant prior memory for the query."""
    blocks = recall_blocks(settings, user_id, session_id, query, k, min_score)
    return ("\n".join(f"- {b['text']}" for b in blocks), len(blocks))


def memory_phase(settings: "config.Settings", session_id: str, prompt: str,
                 min_score: Optional[float] = None, hit_threshold: Optional[float] = None) -> dict:
    """Memory-first lookup for a RAG turn. Returns a dict:
      enabled, blocks, top_score, hit (bool), hit_answer (str|None), context (str), recall_seconds.
    A 'hit' means the top memory is essentially the same question (score >= hit_threshold) and has a
    stored answer — the caller can short-circuit the vector search + LLM and answer from memory."""
    import time
    result = {"enabled": settings.ai_dataplane_enabled, "blocks": [], "top_score": 0.0,
              "hit": False, "hit_answer": None, "context": "", "recall_seconds": 0.0}
    if not settings.ai_dataplane_enabled:
        return result
    if hit_threshold is None:
        hit_threshold = DEFAULT_HIT_THRESHOLD
    t0 = time.time()
    blocks = recall_blocks(settings, DEFAULT_USER, session_id, prompt, min_score=min_score)
    result["recall_seconds"] = time.time() - t0
    result["blocks"] = blocks
    if blocks:
        result["top_score"] = blocks[0]["score"]
        result["context"] = build_memory_context(blocks)
        top = blocks[0]
        if top["score"] >= hit_threshold and top["assistant_content"].strip():
            result["hit"] = True
            result["hit_answer"] = top["assistant_content"].strip()
    return result


# Cosine at/above which a recalled memory is strong enough to use as real context. Below this it's
# shown to the model but explicitly flagged low-relevance, so a different-topic question doesn't get
# misled by a weakly-related prior turn. 0.7 keeps same-topic recalls "relevant" while borderline
# cross-topic ones (e.g. "movies about escapes" vs "movies about cars" ≈ 0.63 with MiniLM) read as
# low relevance. Override with AGENTMEMORY_STRONG_SCORE.
AUGMENT_STRONG_SCORE = float(os.getenv("AGENTMEMORY_STRONG_SCORE", "0.7"))


def build_memory_context(blocks: List[dict]) -> str:
    """Render recalled memory with per-item relevance scores and an instruction to weight by
    relevance — so weakly-related memories are visible but discounted rather than treated as fact."""
    lines = []
    for b in blocks:
        score = b["score"]
        tag = "relevant" if score >= AUGMENT_STRONG_SCORE else "low relevance — likely unrelated"
        lines.append(f"- [relevance {score:.2f} — {tag}] {b['text']}")
    guidance = ("The following are recalled memories from earlier in this conversation, each with a "
                "relevance score (0-1). Use high-relevance items as context; treat low-relevance "
                "items as probably unrelated and ignore them unless they clearly help. Rely on the "
                "retrieved documents below for the actual answer.")
    return guidance + "\n" + "\n".join(lines)


def remember(settings: "config.Settings", user_id: str, session_id: str,
             user_content: str, assistant_content: str) -> bool:
    """Store a conversation turn as memory. Returns True on success, False on any error."""
    try:
        with _client(settings) as client:
            ensure_session(client, user_id, session_id)
            sess = client.get_user(user_id).get_session(session_id)
            sess.add_memory(messages=[{
                "user_content": user_content or "",
                "assistant_content": assistant_content or "",
            }])
            return True
    except Exception:  # noqa: BLE001
        return False


def memory_health(settings: "config.Settings") -> Tuple[bool, str]:
    """(active, detail) for the agent-memory server via the SDK's health check."""
    try:
        with _client(settings) as client:
            status = client.health_ping().overall_status.value
            return (status == "healthy", f"server healthy ({status})")
    except Exception as e:  # noqa: BLE001
        return (False, f"unreachable at {settings.agent_memory_base_url}: {e}")


def memory_stats(settings: "config.Settings") -> dict:
    """Live usage counters from the memory server's /stats endpoint ({} on error)."""
    try:
        import httpx
        r = httpx.get(settings.agent_memory_base_url.rstrip("/") + "/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return {}


def status(settings: "config.Settings") -> dict:
    """Aggregate status for the diagram overlay: toggle + memory active/used/tokens-served."""
    active, detail = memory_health(settings)
    stats = memory_stats(settings) if active else {}
    return {
        "enabled": settings.ai_dataplane_enabled,
        "capabilities": {
            "agent_memory": {
                "active": active,
                "detail": detail,
                "base_url": settings.agent_memory_base_url,
                "used": bool(stats.get("memory_added") or stats.get("searches")),
                "memory_added": stats.get("memory_added", 0),
                "searches": stats.get("searches", 0),
                "blocks_recalled": stats.get("blocks_recalled", 0),
                "tokens_served_from_memory_est": stats.get("tokens_served_from_memory_est", 0),
            },
        },
    }
