#!/usr/bin/env python3
"""
Interactive or CLI-driven RAG demo using Couchbase Hybrid Vector Queries.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator

from couchbase.search import SearchRequest, GeoDistanceQuery
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.exceptions import DocumentNotFoundException

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

    latitude: float
    longitude: float
    radius: str

    index_name: str

    embedding_field: str = "embedding"
    content_field: str = "contents"
    limit: int = 5
    num_candidates: Optional[int] = None  # FTS recall knob (nprobes analogue); None → use limit


def load_config_from_args(args, inputs) -> RAGConfig:
    return RAGConfig(
        bucket=args.bucket,
        scope=args.scope,
        collection=args.collection,
        latitude=float(inputs["latitude"]),
        longitude=float(inputs["longitude"]),
        radius=inputs["radius"],
        index_name=args.index_name,
        limit=args.limit,
        num_candidates=args.num_candidates,
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

    resp = client.chat.completions.create(
        model=model,
        messages=[
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
        ],
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------
# Couchbase
# ------------------------------------------------------------

def get_cluster():
    return Cluster(
        CFG.couchbase_connstr,
        ClusterOptions(
            PasswordAuthenticator(
                CFG.couchbase_username,
                CFG.couchbase_password,
            )
        ),
    )


def run_hybrid_query(cluster, cfg: RAGConfig, query_embedding: List[float]):
    scope = cluster.bucket(cfg.bucket).scope(cfg.scope)
    collection = scope.collection(cfg.collection)

    geo_filter = GeoDistanceQuery(
        location=(cfg.longitude, cfg.latitude),  # (lon, lat)
        distance=cfg.radius,
        field="location",
    )

    vector_query = VectorQuery.create(
        field_name=cfg.embedding_field,
        vector=query_embedding,
        num_candidates=cfg.num_candidates if cfg.num_candidates is not None else cfg.limit,
        prefilter=geo_filter,
    )

    vector_search = VectorSearch.from_vector_query(vector_query)
    request = SearchRequest.create(vector_search)

    search_result = scope.search(cfg.index_name, request)

    hits = [{"id": hit.id, "score": hit.score} for hit in search_result.rows()]

    if not hits:
        return []

    enriched = []

    for h in hits:
        try:
            res = collection.get(h["id"])
            doc = res.content_as[dict]
        except DocumentNotFoundException:
            continue

        enriched.append(
            {
                "id": h["id"],
                "score": h["score"],
                "name": doc.get("name"),
                "contents": doc.get(cfg.content_field),
            }
        )

    return enriched


# ------------------------------------------------------------
# CLI / Input Handling
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Vector RAG Demo")

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--collection", required=True)

    parser.add_argument("--index-name", default="ix-yelp-business-vector")
    parser.add_argument("--limit", type=int, default=25)

    parser.add_argument("--prompt")
    parser.add_argument("--latitude")
    parser.add_argument("--longitude")
    parser.add_argument("--radius")

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
        "--num-candidates",
        type=int,
        default=None,
        help="FTS vector num_candidates — the recall/latency knob (analogue of IVF nprobes). "
             "Higher = better recall, slower. Omit to default to --limit.",
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
        missing = [x for x in ("latitude", "longitude", "radius") if getattr(args, x) is None]
        if missing:
            LOG.error("When using CLI mode, --latitude, --longitude, and --radius are required")
            sys.exit(1)

        return {
            "prompt": args.prompt,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "radius": args.radius,
        }

    prompt = input("\nEnter your prompt:\n> ").strip()
    latitude = input("\nEnter latitude:\n> ").strip()
    longitude = input("\nEnter longitude:\n> ").strip()
    radius = input("\nEnter radius (e.g. 5mi):\n> ").strip()

    if not prompt or not latitude or not longitude or not radius:
        print("Prompt, latitude, longitude, and radius are required")
        sys.exit(1)

    return {
        "prompt": prompt,
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== Couchbase Hybrid Vector RAG Demo ===\n")

    args = parse_args()

    global CFG
    CFG = config.Settings.load(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimensions=args.dimensions,
    )
    CFG.validate_for_rag()

    inputs = resolve_inputs(args)

    cfg = load_config_from_args(args, inputs)
    cluster = get_cluster()

    prompt = inputs["prompt"]
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

    LOG.info("Running hybrid vector query")
    t = time.time()
    chunks = run_hybrid_query(cluster, cfg, embedding)
    timings["vector query"] = time.time() - t

    if not chunks:
        print("\nNo matching context found.")
        return

    formatted_chunks = []
    for c in chunks:
        if c.get("contents"):
            formatted_chunks.append(
                f"Name: {c.get('name', 'Unknown')}\n\n{c['contents']}"
            )

    if not formatted_chunks:
        print("\nNo usable context found.")
        return

    context = "\n\n---\n\n".join(formatted_chunks)
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