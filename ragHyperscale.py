#!/usr/bin/env python3
"""
Interactive or CLI-driven RAG demo using Couchbase Hyperscale Vector Queries.

Rules:
- --bucket, --scope, --collection are REQUIRED
- If ANY user-input CLI args are provided, ALL must be provided
- Otherwise, user inputs are collected interactively
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

load_dotenv()

import config
import dataplane

# Only force HF fully offline when no HF_TOKEN is provided. With a token, allow authenticated
# online access (higher rate limits, #6); without one, offline avoids the per-run cache
# revalidation HEAD storm that triggers HTTP 429 backoffs.
if not os.getenv("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOG = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ------------------------------------------------------------
# OpenAI Client (OpenAI-compatible)
# ------------------------------------------------------------

_openai_client = None


def get_openai_client() -> OpenAI:
    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=CFG.openai_api_key,
            base_url=CFG.openai_base_url,
        )

    return _openai_client


# ------------------------------------------------------------
# Dataset / Query Configuration
# ------------------------------------------------------------

@dataclass
class RAGConfig:
    bucket: str
    scope: str
    collection: str

    embedding_field: str = "embedding"
    content_field: str = "contents"

    limit: int = 5
    nprobes: int = 3  # IVF probe count (recall/latency knob)


def load_config_from_args(args) -> RAGConfig:
    return RAGConfig(
        bucket=args.bucket,
        scope=args.scope,
        collection=args.collection,
        nprobes=args.nprobes,
    )


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

_st_model = None

# Central config (backlog #3): single source of truth for provider/model/dimension + creds.
# Populated in main() from .env + CLI overrides.
CFG: "config.Settings" = None


def resolve_provider() -> str:
    """Effective embedding provider, from the central config."""
    return CFG.provider


def st_model_name() -> str:
    """Effective local embedding model name, from the central config."""
    return CFG.embedding_model


def openai_embedding_model() -> str:
    """Effective OpenAI embedding model name, from the central config."""
    return CFG.embedding_model


def compute_embedding(text: str) -> List[float]:
    """
    Compute an embedding using the resolved provider:
    - "local"  → SentenceTransformer (falls back to OpenAI only on error)
    - "openai" → OpenAI-compatible API
    """
    global _st_model

    if resolve_provider() == "local":
        name = st_model_name()
        try:
            if _st_model is None:
                LOG.info("Loading SentenceTransformer: %s", name)
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(name)

            return _st_model.encode(text).tolist()
        except Exception as e:
            LOG.warning("SentenceTransformer failed, falling back to OpenAI: %s", e)

    client = get_openai_client()
    model = openai_embedding_model()

    resp = client.embeddings.create(
        model=model,
        input=text,
    )

    return resp.data[0].embedding


def preflight_dimensions(actual_dim: int, collection: str) -> None:
    """Fail fast when the embedding dimension disagrees with VECTOR_DIMENSIONS (.env).
    A mismatch (e.g. a 1536-dim query against a 384-dim collection) otherwise returns an
    empty result with no error — this turns that silent miss into an actionable message."""
    declared = CFG.declared_dimensions
    if declared is None:
        LOG.info("Preflight skipped: no expected dimension (set VECTOR_DIMENSIONS or --dimensions).")
        return
    if actual_dim != declared:
        LOG.error(
            "Embedding dimension mismatch: provider '%s' model '%s' produced %d-dim vectors but "
            "expected %d (querying collection '%s'). The provider/model, the collection's vector "
            "index, and the expected dimension must all agree (local/MiniLM=384, OpenAI=1536).",
            CFG.provider, CFG.embedding_model, actual_dim, declared, collection,
        )
        sys.exit(1)
    LOG.info("Preflight OK: %d-dim %s embeddings match expected dimension (collection '%s').",
             actual_dim, CFG.provider, collection)


# ------------------------------------------------------------
# LLM Generation
# ------------------------------------------------------------

def generate_with_llm(prompt: str, context: str) -> str:
    client = get_openai_client()
    model = CFG.openai_chat_model

    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the provided context. "
                "If the context is insufficient, say so plainly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nPrompt:\n{prompt}",
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------
# Couchbase
# ------------------------------------------------------------

def get_cluster():
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions
    from couchbase.auth import PasswordAuthenticator

    return Cluster(
        CFG.couchbase_connstr,
        ClusterOptions(
            PasswordAuthenticator(
                CFG.couchbase_username,
                CFG.couchbase_password,
            )
        ),
    )


def run_hyperscale_query(
    cluster,
    cfg: RAGConfig,
    query_embedding: List[float],
) -> List[str]:
    params = {
        "vector": query_embedding,
        "limit": cfg.limit,
    }

    statement = f"""
    SELECT RAW 'Title: ' || m.Title || ': ' || m.{cfg.content_field}
    FROM `{cfg.bucket}`.`{cfg.scope}`.`{cfg.collection}` m
    ORDER BY APPROX_VECTOR_DISTANCE(m.{cfg.embedding_field}, $vector, "COSINE", {cfg.nprobes})
    LIMIT $limit
    """

    print("\n=== Query ===\n")
    print(statement, {**params, "vector": "<removed>"})

    return [row for row in cluster.query(statement, **params)]


# ------------------------------------------------------------
# CLI / Input Handling
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperscale Vector RAG Demo")

    # Required dataset args
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--collection", required=True)

    # Optional user-input args
    parser.add_argument("--prompt")

    # Embedding provider override (see #1). Omit to keep the .env-driven default.
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai"],
        default=None,
        help="Force the embedding provider. Omit to infer from .env "
             "(SENTENCE_TRANSFORMER_MODEL set → local, else openai). Must match the "
             "collection's index dimensions (local/MiniLM=384, openai=1536).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model name for the chosen provider (local sentence-transformers "
             "model or OpenAI embedding model). Its output dimension must match the collection's index.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Override the expected embedding dimension for the preflight check. Precedence: this "
             "flag > VECTOR_DIMENSIONS (.env) > derived from the model.",
    )
    parser.add_argument(
        "--nprobes",
        type=int,
        default=3,
        help="IVF probe count for APPROX_VECTOR_DISTANCE (recall vs latency). Higher = more index "
             "cells scanned = better recall, slower. Needs enough loaded data to show a difference.",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Agent-memory session id — groups a conversation. Only used when the AI Data Plane "
             "is enabled (AI_DATAPLANE_ENABLED); recalls prior turns and remembers this one.",
    )
    parser.add_argument(
        "--memory-min-score",
        type=float,
        default=None,
        help="Minimum cosine relevance (0-1) for recalling a memory. Higher = stricter; repeats and "
             "closely-related questions still match. Omit to use the memory server's default.",
    )
    parser.add_argument(
        "--memory-hit-threshold",
        type=float,
        default=None,
        help="Cosine at/above which a recalled memory is treated as the SAME question and answered "
             "from memory, skipping the vector search + LLM (default ~0.9).",
    )

    return parser.parse_args()


def resolve_inputs(args):
    if args.prompt:
        return args.prompt

    prompt = input("\nEnter your prompt:\n> ").strip()

    if not prompt:
        print("Prompt required")
        sys.exit(1)

    return prompt


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== Couchbase Hyperscale Vector RAG Demo ===\n")

    args = parse_args()

    global CFG
    CFG = config.Settings.load(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimensions=args.dimensions,
    )
    CFG.validate_for_rag()

    prompt = resolve_inputs(args)

    cfg = load_config_from_args(args)
    cluster = get_cluster()

    timings = {}

    # Memory-first: a near-duplicate question is answered from memory, skipping vector search + LLM.
    mem = dataplane.memory_phase(CFG, args.session, prompt,
                                 min_score=args.memory_min_score,
                                 hit_threshold=args.memory_hit_threshold)
    if mem["enabled"]:
        timings["memory recall"] = mem["recall_seconds"]
        LOG.info("AI Data Plane: recalled %d block(s), top score %.2f (session '%s')",
                 len(mem["blocks"]), mem["top_score"], args.session)

    if mem["hit"]:
        print("\n=== Answer (from memory) ===\n")
        print(mem["hit_answer"])
        print(f"\n⚡ Served from memory in {mem['recall_seconds']:.2f}s (top score {mem['top_score']:.2f}) — "
              f"skipped the vector search + LLM call (~{dataplane.est_tokens(mem['hit_answer'])} tokens of LLM output avoided).")
        return

    LOG.info("Computing embedding")
    t = time.time()
    embedding = compute_embedding(prompt)
    timings["embed"] = time.time() - t
    preflight_dimensions(len(embedding), cfg.collection)

    LOG.info("Running hyperscale vector query")
    t = time.time()
    chunks = run_hyperscale_query(cluster=cluster, cfg=cfg, query_embedding=embedding)
    timings["vector query"] = time.time() - t

    if not chunks:
        print("\nNo matching context found.")
        return

    context = "\n\n---\n\n".join(chunks)
    if mem["context"]:
        context = f"{mem['context']}\n\n--- Retrieved documents ---\n\n{context}"

    print("\n=== Context to Augment with ===\n")
    print(context)

    t = time.time()
    answer = generate_with_llm(prompt, context)
    timings["LLM"] = time.time() - t

    print("\n=== Answer ===\n")
    print(answer)

    dataplane.print_timings(timings, extra=f"LLM input ~{dataplane.est_tokens(context + prompt)} tokens (est)")
    dataplane.maybe_remember(CFG, args.session, prompt, answer)


if __name__ == "__main__":
    main()