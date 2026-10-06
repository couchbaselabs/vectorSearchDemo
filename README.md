# Vector Search Indexes - Couchbase Vector Query Options in action

This demo shows three different ways to run vector search in Couchbase—**Hyperscale**, **Composite**, and **Hybrid (FTS)**, using real datasets and the same end-to-end flow. You’ll load data, generate embeddings, create the appropriate index, and run a RAG-style query to see how each approach affects relevance, filtering, and query flexibility. Each section uses a different dataset to highlight when one vector search option is a better fit than the others.

In this demo, you will:
- Load real-world datasets and generate embeddings
- Create Hyperscale, Composite, and Hybrid vector indexes
- Run RAG-style queries against each index type
- Compare how filtering, relevance, and query expressiveness differ

## Demo map

Each vector search option uses a different dataset and script to highlight when that approach is the best fit:

| Vector option | Dataset | Collection | RAG script |
|--------------|--------|------------|------------|
| Hyperscale | Wikipedia movie plots | `movies` | `ragHyperscale.py` |
| Composite | Customer care emails | `emails` | `ragComposite.py` |
| Hybrid (FTS) | Yelp businesses | `yelp` | `ragHybrid.py` |

(Use the CLI if you want to see the exact queries and tweak parameters; use the UI if you want a faster, more guided way to explore the same workflows).

## Success looks like this

After running this demo, you should be able to see and explain:

- **Hyperscale**: Broad semantic similarity across a large dataset, with no structured filtering.
- **Composite**: More precise results by combining semantic similarity with structured filters like sender, case ID, or product.
- **Hybrid (FTS)**: The ability to mix semantic search with search engine features such as keywords, text, and geospatial constraints.

If you can clearly describe *why* a given query uses one index type over the others, the demo is working as intended.

# Three Types of Vector Search indexes

## Hyperscale

This type of index is best for large data sets where you don't plan to do any filtering of data.

Example use cases:

- **Knowledge base**: A collection of car owner's manuals: What are the most common causes of muffler failure in vehicles?
- **Coding copilot**: examine all code repositories with a similar purpose to help generate a function
- **Research/writing**: examine all books for a related topic to generate a paragraph

## Composite

A hyperscale vector search (k-NN over all vectors) finds semantically similar documents but cannot efficiently restrict the search by structured attributes (for example, author or recipient of an email). A composite vector query lets you apply standard filters (e.g., `sender == X`) and then run a k-NN search only within that filtered subset, reducing noise and improving both accuracy and performance.

Example use cases
- **Email content** What organizations do emails from `jim.smith@gmail.com` mention the most?
- **Legal eDiscovery**: Restrict by `case_id` then run vector search to find semantically relevant clauses or communications.
- **Customer Support Triage**: Filter by product and retrieve semantically similar past tickets/solutions.
- **Medical Records Retrieval**: Limit by `patient_id` and perform semantic retrieval over clinical notes.

## Hybrid (FTS)

A hybrid vector search can use semantic vector search together with FTS features (for example, geospatial or traditional text).

Example use cases:
- **Business search**: What businesses within 5 miles of a certain location might help me with weight loss? (geospatial + vector)
- **Job search**: Find software jobs mentioning `C#` and `cloud`, ranked by semantic similarity to `backend API development`, posted in the last 30 days. (keyword + vector + range)
- **Content moderation**: Locate social posts mentioning a specific event or location, ranked by semantic similarity to harassment or threats. (keyword + vector)
- **News analysis**: List articles mentioning `interest rates` or `inflation`, ranked by similarity to recession risk narratives. (keyword + vector)

# Step 0: Prerequisites

## Install Python

This demo requires **Python 3.12 or newer**. Check what you have with `python3 --version` (macOS/Linux) or `python --version` (Windows).

If you need to install or upgrade Python, use your platform's package manager:

**macOS (Homebrew):**

```bash
# Install Homebrew first if you don't have it: https://brew.sh
brew install python@3.12
# brew installs it as `python3.12`; confirm:
python3.12 --version
```

**Windows (winget):**

```powershell
winget install Python.Python.3.12
# Open a new terminal, then confirm:
python --version
```

On Linux, use your distro's package manager (e.g. `sudo apt install python3.12 python3.12-venv` on Debian/Ubuntu).

## Create a virtual environment

Once Python is installed, create a virtual environment (use the `python3.12` binary if `python3` points at an older version):

```bash
# Linux/macOS
python3 -m venv venv

# Windows
python -m venv venv
```

NOTE: Creating a venv is a one-time operation.

Then go into that venv with this command:

```bash
# Linux
source venv/bin/activate

# Windows PowerShell
# you may need `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first
.\venv\Scripts\Activate.ps1
```

Then install requirements:

```bash
python -m pip install -r requirements.txt
```

You'll need an `.env` file holding your Couchbase connection string, credentials, and (for the RAG scripts) an OpenAI key. You have three ways to create/edit it — pick whichever you prefer; they all read and write the same `.env`. See **Configuring the demo** below for details. You'll get the actual connection string and credentials from the Capella cluster you set up in **Step 0.5**.

Now you're ready to set up your Couchbase Capella cluster.

# Step 0.5: Set up Couchbase Capella

Before you can load any data, you need a Couchbase Capella cluster with a bucket, the collections this demo uses, a database user, and network access configured. If you don't have a cluster yet, create a free tier one from the [Capella UI](https://cloud.couchbase.com/) first.

## 1. Create a bucket

In the Capella UI, open your cluster and go to **Data Tools → Buckets** (or **Settings → Buckets**), then **Create Bucket**:

- **Name:** `vectorSearchDemo` (this is what the demo commands use; use another name if you prefer, but pass it consistently via `--bucket`).
- Leave the memory quota at the default for the free tier.

> Bucket creation isn't available through SQL++/N1QL — it must be done in the Capella UI (or the Management API).

## 2. Create the collections

Every new bucket has a `_default` scope, but this demo stores each dataset in its own **collection** under that scope, and those collections do **not** exist by default. Create them.

Easiest path — open the **Query Workbench** (Data Tools → Query) and run:

```sql
CREATE COLLECTION `vectorSearchDemo`.`_default`.`movies` IF NOT EXISTS;
CREATE COLLECTION `vectorSearchDemo`.`_default`.`emails` IF NOT EXISTS;
CREATE COLLECTION `vectorSearchDemo`.`_default`.`yelp`   IF NOT EXISTS;
```

(You can also create them via **Data Tools → Collections** in the UI.) If you plan to use the OpenAI (1536-dim) provider, also create the parallel `movies_openai` / `emails_openai` / `yelp_openai` collections.

To run ad-hoc `SELECT`s for troubleshooting (the vector indexes don't serve plain selects), also add a primary index per collection:

```sql
CREATE PRIMARY INDEX ON `vectorSearchDemo`.`_default`.`movies`;
CREATE PRIMARY INDEX ON `vectorSearchDemo`.`_default`.`emails`;
CREATE PRIMARY INDEX ON `vectorSearchDemo`.`_default`.`yelp`;
```

## 3. Create a database user with a custom role

The demo scripts connect with a Couchbase **database user** (this is separate from your Capella login). Go to **Settings → Cluster Access → Create Database Credentials** and create one:

- **Username / password:** choose your own; these go into `.env` as `COUCHBASE_USERNAME` / `COUCHBASE_PASSWORD`.
- **Roles:** grant a custom role scoped to the demo bucket with **all grants** — the simplest reliable option is to give the credential **read and write access to all buckets** (or at minimum to `vectorSearchDemo`), covering Data, Query, and Search. The loader writes documents, the RAG scripts query the Data/Query/Search services, and index creation needs the management grants, so the credential needs the full set of data + query + search privileges on the bucket.

> If you scope the role too narrowly (e.g. read-only, or Data but not Query/Search), you'll see authentication or "permission denied" errors when loading or querying. When in doubt for a demo, grant full access to `vectorSearchDemo`.

## 4. Allow network access

Capella blocks all inbound connections by default. Go to **Settings → Networking → Allowed IP Addresses → Add Allowed IP** and either:

- Add your current IP, or
- **Allow access from anywhere** by adding `0.0.0.0/0`.

> `0.0.0.0/0` opens the cluster to the entire internet. That's fine for a short-lived, throwaway demo cluster, but don't leave it on a cluster with real data — restrict it to known IPs and remove the rule when the demo is over.

## 5. Get the connection string

Go to the **Connect** tab in the Capella UI (**cluster → Connect**). Copy the **Public Connection String** shown there — it looks like `couchbases://cb.xxxxxxxx.cloud.couchbase.com`.

Put it, along with the database user from step 3, into your `.env`:

```bash
COUCHBASE_CONNSTR=couchbases://cb.xxxxxxxx.cloud.couchbase.com
COUCHBASE_USERNAME=your-db-user
COUCHBASE_PASSWORD=your-db-password
```

Now you're ready to configure the demo and start loading data.

# Configuring the demo

All configuration lives in a single `.env` file (Couchbase connection + credentials, embedding provider/model, OpenAI key). There are three ways to create and edit it — they all read and write the same `.env`, so you can mix and match:

## Option A — Setup wizard (terminal, no manual editing)

Run the interactive wizard and answer the prompts:

```bash
python setup.py
```

It shows your current values as defaults (so re-running only changes what you type), writes `.env` for you, and offers to run the connectivity test at the end. This is the easiest option if you'd rather not touch `.env` by hand.

## Option B — Web UI configuration drawer

Start the web UI (`python app.py`) and click **⚙ Configuration** in the top-right. The drawer lets you enter the connection string, credentials, embedding provider/model, and OpenAI key, then:

- **Test connection** runs the same readiness checks below against the values in the form (before saving).
- **Save to .env** writes them to `.env`.

Existing secrets (password, API key) are shown masked; leave them as-is to keep the stored value. Saving normalizes `.env` formatting (consistent key order; inline comments are dropped), but never changes values you didn't edit.

## Option C — Edit `.env` by hand

Copy the sample and edit it in your editor of choice:

```bash
cp .env.sample .env      # Linux/macOS
copy .env.sample .env    # Windows
```

`.env.sample` documents every setting (connection, provider/model, the 384/1536 dimension matrix, optional `HF_TOKEN`).

## Test connectivity before loading

However you configured it, verify everything is reachable **before** loading data — cluster connection, bucket + collections, the embedding provider's dimension, and the OpenAI key:

```bash
python setup.py    # answer "y" at the connectivity-test prompt, or just re-run and keep existing values
```

or click **Test connection** in the web UI drawer. A passing run looks like:

```
✓ Couchbase connection: reachable at couchbases://cb.xxxx.cloud.couchbase.com
✓ Bucket: 'vectorSearchDemo' found
✓ Collections: found under _default: ['movies', 'emails', 'yelp']
✓ Embedding provider: provider 'local' model 'all-MiniLM-L6-v2' → 384-dim (matches expected 384)
✓ OpenAI (for RAG): key valid; chat model 'gpt-4o-mini'
```

A failed check tells you exactly what to fix (unreachable cluster → check the connection string / allowed IPs from Step 0.5; missing collections → create them; dimension mismatch → provider/model doesn't match the index).

> The web UI writes secrets to `.env` on Save and executes commands locally with no authentication — only run it on `localhost` for a trusted, local demo.

# (Optional) Set up Agent Memory — AI Data Plane

> This is only needed for the **AI Data Plane** features of the demo (agent memory on top of RAG). You can skip it and run the plain RAG demo without it. A Capella-hosted Agent Memory endpoint is coming; until then you run the server locally alongside `app.py`.

This repo ships a lightweight, **native** Agent Memory server (`memory_server.py`) — no Docker. It runs next to `app.py`, stores its data in your Couchbase cluster, and is driven by the real `couchbase-agent-memory` SDK (`AgentMemoryClient`). The app talks to it over HTTP (`AGENT_MEMORY_BASE_URL`, default `http://localhost:8090` — 8090 because the web UI uses 8080).

> The native server implements the SDK's user/session/memory API and recalls memories by vector similarity on OpenAI embeddings. It intentionally does **not** replicate the official server's background LLM "fact extraction" — it stores conversation turns and recalls the most relevant ones. When the Couchbase-hosted Agent Memory endpoint is available, just point `AGENT_MEMORY_BASE_URL` at it — the same SDK calls work unchanged. An optional path for running the official container instead is at the end of this section.

## 1. Create its scope and collections in the cluster

The server uses a fixed scope `agentmemory` with three collections. Create them in your bucket (Query Workbench):

```sql
CREATE SCOPE `vectorSearchDemo`.`agentmemory` IF NOT EXISTS;
CREATE COLLECTION `vectorSearchDemo`.`agentmemory`.`users`    IF NOT EXISTS;
CREATE COLLECTION `vectorSearchDemo`.`agentmemory`.`sessions` IF NOT EXISTS;
CREATE COLLECTION `vectorSearchDemo`.`agentmemory`.`memory`   IF NOT EXISTS;
```

## 2. Run the native server

It reads `AGENTMEMORY_*` from `.env` and falls back to your `COUCHBASE_*` / `OPENAI_API_KEY`, so if your `.env` is already configured there's nothing extra to set:

```bash
python memory_server.py          # serves http://localhost:8090
```

## 3. Verify it's healthy

```bash
curl -s http://localhost:8090/health
```

or from Python (the real SDK):

```python
from agentmemory import AgentMemoryClient   # pip install couchbase-agent-memory
with AgentMemoryClient(base_url="http://localhost:8090") as client:
    print(client.health_ping().overall_status.value)   # -> healthy
```

The app picks it up via `AGENT_MEMORY_BASE_URL` in `.env` (default `http://localhost:8090` already matches).

## (Optional) Use the official Agent Memory container instead

If you have access to the official `agentmemory-server` image (obtained via your Couchbase download portal — it's a gated artifact, `docker load`ed from a tarball, not a public pull) and want the full server with background fact-extraction, run it instead of `memory_server.py`. It also reads the `AGENTMEMORY_*` variables and listens on 8080 internally:

```bash
docker run -d --name agentmemory-server --env-file .env \
  -p 8090:8080 -p 9090:9090 \
  -v "$(pwd)/ca.pem:/app/certs/ca.pem:ro" \
  --restart unless-stopped agentmemory-server:arm64   # or :amd64
```

Either way, point `AGENT_MEMORY_BASE_URL` at `http://localhost:8090`.

## Using agent memory with RAG

Turn it on with the global toggle (web UI, AI Services tab) or `AI_DATAPLANE_ENABLED=true` in `.env`. When enabled, each RAG query first recalls relevant prior turns and, after answering, remembers the new Q&A. You'll see this in the output:

- **Related question** → recalled memories are prepended to the RAG context, each labeled with a relevance score (0–1) and a strong/low-relevance tag so the model discounts weakly-related ones; the full RAG query still runs.
- **Repeat / near-duplicate question** → answered directly from memory, **skipping the vector search + LLM** — e.g. `⚡ Served from memory in 1.1s — skipped the vector search + LLM call`. (In the web UI this shows as a **⚡ from memory** badge next to Run.)
- Every full run also prints a timing/token breakdown (`memory recall / embed / vector query / LLM / total`).

### Sessions are scoped per workflow

A **session** groups a conversation so memories are recalled within it. The session updates as part of the workflow you choose: each workflow (Hyperscale / Composite / Hybrid) uses its own session (`<base>-hyperscale`, `<base>-composite`, `<base>-hybrid`), so memories from a movies chat aren't recalled during a Yelp chat. In the web UI the active session is shown on the AI Services tab and updates when you switch RAG tabs; **New session** starts a fresh base for all workflows. On the CLI, pass `--session <id>` (default `default`).

### Tuning recall

- **`--memory-min-score <0-1>`** (UI: *Memory recall threshold*): minimum cosine for a memory to be recalled at all. Higher = stricter; repeats and closely-related questions still match. Default comes from the server (`AGENTMEMORY_MIN_SCORE`, 0.2).
- **`--memory-hit-threshold <0-1>`**: cosine at/above which a memory is treated as the *same* question and answered from memory (short-circuit). Default ~0.9.
- **`AGENTMEMORY_STRONG_SCORE`** (default 0.7): at/above this a recalled memory is labeled "relevant"; below it is shown but flagged "low relevance" so the model discounts it.
- Failed answers ("the context doesn't contain that") are **not** cached, so a later repeat re-runs RAG instead of replaying a non-answer.

# Step 1: Loading the data

The data must first be loaded into Couchbase. The `load.py` script will load data into Couchbase, giving them embeddings with the specified model (configuration in `.env`).

You can load with a command like this:

```bash
python load.py --data data.json --id-field id --text-fields contents --bucket mybucket --scope myschema --collection mydocs --limit 5
```

The `--limit N` parameter means that you want to load the next N documents that haven't been loaded yet.

## Choosing the embedding provider

By default the embedding provider is inferred from `.env`: if `SENTENCE_TRANSFORMER_MODEL` is set, `load.py` and the `rag*.py` scripts use a local sentence-transformers model (e.g. `all-MiniLM-L6-v2`, **384 dimensions**); otherwise they use OpenAI (`text-embedding-3-small`, **1536 dimensions**).

You can override this per-invocation with `--embedding-provider {local,openai}` on `load.py` and all three `rag*.py` scripts — handy for switching between a 384-dim and a 1536-dim collection without editing `.env` mid-demo:

```bash
python load.py ... --collection yelp        --embedding-provider local
python load.py ... --collection yelp_openai --embedding-provider openai
```

You can also override the model name itself with `--embedding-model <name>` (the local sentence-transformers model or the OpenAI embedding model), and the expected dimension with `--dimensions <n>`. Its output dimension must still match the target index.

`load.py` is idempotent (it skips document IDs that already exist). When you switch embedding model or provider on a collection that already has data, pass `--overwrite` so the existing rows are re-embedded and replaced rather than skipped — otherwise stale old-dimension vectors remain.

Configuration is centralized in `config.py` (a small pydantic settings module): it reads `.env`, validates per-provider requirements, and resolves the effective provider/model/dimension with the precedence documented below.

> **Dimensions must agree.** The provider you load with, the vector index's `dimension`, and the provider you query with must all match (local = 384, OpenAI = 1536). A mismatch returns an empty result ("No matching context found") with no error. When you override the provider, point the command at a collection/index built for that dimension. As a safety net, the scripts run a startup preflight that fails fast if the embedding dimension disagrees with `VECTOR_DIMENSIONS` in `.env`.

In the web UI (`app.py`), a single **Embedding provider / Model** control at the top drives every load and query command: it appends the flags and maps each dataset to its local collection or its `<name>_openai` (1536-dim) collection (and swaps the Hybrid FTS index to match).

When stored in Couchbase as a document with an embedding, documents will look like this:

```javascript
[
    key: "doc1"
    {
        "emailFrom": ["bobsmith@gmail.com"],
        "caseId": "123456",
        ... etc ...
        "textToUseInVector": "...text goes here...",
        "embedding": [0.0123, -0.8471, ... etc ...]
    },
    // ... etc ...
]
```

Alternatively, the embeddings can be automatically generated on the fly with Capella AI Services with a [Process and Vectorize Unstructed Data workflow](https://docs.couchbase.com/ai/build/vectorization-service/vectorize-structured-data-capella.html). This approach will greatly simplify your AI application development, and separates the data processing from your application code.

When embedding with AI Services, that "embedding" field would be automatically created/updated whenever the document itself is created/updated. Furthermore, AI Services can use either an external Open AI type of model, or a private model hosted in Capella itself. (A private model could also be used in `load.py`).

Here are three examples of loading data for the three use cases:

**Composite** - load emails from the [Customer Care Emails dataset](https://www.kaggle.com/datasets/rtweera/customer-care-emails): dataset.csv

```bash
python load.py --data data/dataset.csv --text-fields subject message_body --bucket vectorSearchDemo --scope _default --collection emails --copy-fields subject sender receiver message_body --limit 5 --id-field sender timestamp
```

> NOTE: Two fields being vectorized, combined into one "content" field. I also have both of them in `--copy-fields` so they can stay separate. Your needs will vary by use case.

This will vectorize the subject+message body together, and save the sender and receiver for filtering. It also combines sender/timestamp combination as a unique ID for each document.

**Hyperscale** - load movie plots from the [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots): wiki_movie_plots_deduped.csv

```bash
python load.py --data data/wiki_movie_plots_deduped.csv --text-fields Plot --bucket vectorSearchDemo --scope _default --collection movies --copy-fields Title "Release Year" Director --limit 5 --id-field Title "Release Year"
```

> NOTE: The CSV header is `Release Year` (with a space), so it must be quoted (`"Release Year"`) or escaped (`Release\ Year`) on the command line — otherwise `--id-field` can't find it and every row is skipped. Since `Plot` is the single field being vectorized, it's the only `--text-fields` value; the others are carried along via `--copy-fields`. Your needs will vary by use case.

**Hybrid** - load businesses from the [Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset): yelp_academic_dataset_business.json

```bash
python load.py --data data/yelp_academic_dataset_business.json --text-fields categories --bucket vectorSearchDemo --scope _default --collection yelp --copy-fields latitude longitude name --limit 5 --id-field business_id
```

# Step 2: Create the indexes

Once the data is loaded, create index(es).

## [Composite Vector Index](https://docs.couchbase.com/cloud/vector-index/composite-vector-index.html).

```SQL
CREATE INDEX `idx_comp_vector_email`
ON `emails`(`embedding` VECTOR,`sender`,`receiver`)
WITH {  "dimension":384, "similarity":"DOT", "description":"IVF,SQ8" }
```

Important notes:

* `DOT` similarity is used because it's good for comparing text content.
* Make sure the number of dimensions matches your .env setting.

Test the index with a query like:

```SQL
WITH anEmail AS (
    SELECT RAW embedding
    from `vectorSearchDemo`.`_default`.`emails` x
    USE KEYS ["Aetheros Support <support@aetheros.com>::2023-10-26T03:42:15Z"]
)
SELECT e.sender, e.receiver, e.content
FROM `vectorSearchDemo`.`_default`.`emails` e
ORDER BY APPROX_VECTOR_DISTANCE(e.embedding,anEmail[0],"DOT")
LIMIT 5;
```

The `WITH` clause here spares us from having to copy/paste a long vector into a sample query. Pick any document key from the data that has been loaded. The result of this query will almost certainly be the email with that timestamp, because it's the most semantically similar. Not a very useful query, but it helps us to verify the index is working.

## [Hyperscale Vector Index](https://docs.couchbase.com/cloud/vector-index/hyperscale-vector-index.html)

```SQL
CREATE VECTOR INDEX `idx_hyperscale_plot`
ON `vectorSearchDemo`.`_default`.`movies`(`embedding` VECTOR)
WITH {
  "dimension": 384,
  "similarity": "COSINE",
  "description": "IVF,SQ8"
};
```

Important notes:
* `COSINE` similarity here is good for text comparison.
* `IVF` with no number allows Capella to choose an appropriate number of centroids
* `dimension` needs to be correct for the model you're using

Test the index with a query like:

```SQL
WITH aMovie AS (
    SELECT RAW m.embedding
    FROM `vectorSearchDemo`.`_default`.`movies` AS m
    WHERE m.Title = "Alice in Wonderland"
    LIMIT 1
)
SELECT m.Title, m.`Release Year`, approx_distance
FROM `vectorSearchDemo`.`_default`.`movies` AS m
LET approx_distance = APPROX_VECTOR_DISTANCE(
    m.embedding, aMovie[0], "COSINE", 3
)
ORDER BY approx_distance
LIMIT 5;
```

## [Hybrid (FTS) Vector Index](https://docs.couchbase.com/cloud/vector-search/vector-search.html)

Import "hybridIndex.json" into a Capella Search index. Note that the number of dimensions must match the model you're using (384 for local sentence transformers, 1536 for OpenAI, etc).

Important notes:
* `cosine` is used because it's good for text comparison

# Step 3: Perform a RAG operation

The `ragXYZ.py` programs are interactive command line programs that allows you to specify a sender and/or receiver(s) email addresses, and enter a prompt. This will be vectorized with the same model as in the load.py scripts. The program will gather the relevant information from the database, using a query corresponding to the index.

## Composite

This is the form of query that will be used for RAG+Composite Query. Note the predicates being used.

```SQL
SELECT RAW e.contents
FROM `vectorSearchDemo`.`_default`.`emails` e
WHERE e.sender == $sender
AND e.receiver == $receiver
ORDER BY APPROX_VECTOR_DISTANCE(e.embedding, <vector of prompt goes here>, "COSINE")
LIMIT 5
```

## Hyperscale

This is the form of query that will be used for RAG+Hyperscale Query. Note that it's entirely based on knn.

```SQL
SELECT RAW m.contents
FROM `vectorSearchDemo`.`_default`.`movies` AS m
LET approx_distance = APPROX_VECTOR_DISTANCE(
    m.embedding, <embedding>, "COSINE", 3
)
ORDER BY approx_distance
LIMIT 5;
```

## Hybrid (FTS)

The Hybrid query (geospatial+vector) is contructed using the Python SDK:

```python
geo_filter = GeoDistanceQuery(
    location=(cfg.longitude, cfg.latitude),  # (lon, lat)
    distance=cfg.radius_miles,
    field="location"
)

vector_query = VectorQuery.create(
    field_name="embedding",
    vector=query_embedding,
    num_candidates=3,
    prefilter=geo_filter
)

vector_search = VectorSearch.from_vector_query(vector_query)
```

This query will ultimately return the best matching documents by ID, along with a score. You can embed content fields in the index, but another common pattern that is often used is to perform KV lookups with the resulting IDs.

The content gathered by these queries will then be sent, along with the prompt, to an LLM (OpenAI gpt-4o-mini), and the result displayed on the command line.

### Tuning recall vs. latency (nprobes / num_candidates)

Each RAG script exposes the vector search's recall/latency knob:

- `ragHyperscale.py` / `ragComposite.py` (GSI): `--nprobes N` — the IVF probe count (the 4th argument of `APPROX_VECTOR_DISTANCE`). Higher = more index cells scanned = better recall, slower. Hyperscale defaults to `3`; Composite omits it (index default) unless you pass the flag.
- `ragHybrid.py` (FTS): `--num-candidates N` — the FTS analogue; defaults to `--limit`.

Higher values only produce a visibly different result once there's enough data for the IVF clusters to differ, so load a larger batch (e.g. `--limit 300`) before demoing the effect. `time python ragHyperscale.py ... --nprobes 1` vs `--nprobes 50` makes the latency tradeoff concrete. All three are also exposed as inputs in the web UI.

## Composite

To execute a RAG prompt with Composite Vector Query:

```bash
python ragComposite.py --bucket vectorSearchDemo --scope _default --collection emails --prompt "Who is mentioned most?" --sender jim@example.com
```

If you don't use `--prompt` then the program will run interactively, asking you for a prompt and filters.

Sample execution:

```bash
== Couchbase Composite Vector RAG Demo ===

Filter by sender (exact match): "Wayne D. Kimmel" <wayne@etfventurefunds.com>
Filter by receiver field (exact match): 

Enter your prompt:
> What organizations does Wayne most discuss?

=== Answer ===

Wayne most discusses ETF Venture Funds and the National Venture Capital Association (NVCA). He also mentions Fisker, indicating an interest in investing in the company.
```

## Hyperscale

To execute a RAG prompt with Hyperscale Query:

```bash
python ragHyperscale.py --bucket vectorSearchDemo --scope _default --collection movies --prompt "What are some movies that involve escapes?"
```

If you don't use `--prompt`, the program will run interactively, asking you for a prompt.

Sample execution:

```bash
=== Couchbase Hyperscale Vector RAG Demo ===

...

=== Query ===

    SELECT RAW 'Title: ' || m.Title || ': ' || m.contents
    FROM `vectorSearchDemo`.`_default`.`movies` m
    ORDER BY APPROX_VECTOR_DISTANCE(m.embedding, $vector, "COSINE", 3)
    LIMIT $limit
     {'vector': '<removed>', 'limit': 5}

=== Context to Augment with ===

Title: For Her Sake: The film is a period drama taking place right before the start of the...
---
Title: The Suburbanite: The film is about a family who move to the suburbs, hoping for a ...
---
Title: The Great Train Robbery: The film opens with two bandits breaking into a railroad telegraph ...
---
Title: The Pasha's Daughter: The film begins with Jack Sparks, a young American, who is traveling...
---
Title: Youth's Endearing Charm: The film is about a court case and embezzlement.

=== Answer ===

The following movies from the provided context involve escapes:

1. **For Her Sake** - The girl helps her lover escape from captivity by giving him a file to free himself from the bars, and they flee on horseback.
2. **The Pasha's Daughter** - Jack Sparks escapes from prison by digging out the bar of his cell window and overpowering a guard before climbing over the wall into the courtyard of the Pasha's palace.

=== finished ===
```

## Hybrid (FTS)

To execute a RAG prompt with Hybrid Query:

```bash
python ragHybrid.py --bucket vectorSearchDemo --scope _default --collection yelp --prompt "Where can I go for various health and wellness services?" --latitude 34.426678 --longitude -119.711196 --radius 5mi
```

If you don't use `--prompt`, `--latitude`, `--longitude`, and `--radius`, the program will run interactively, asking you for all of these data points.

Sample execution:

```bash
=== Couchbase Hybrid Vector RAG Demo ===

...

=== Context to Augment with ===

Name: Abby Rappoport, LAC, CMQ

Doctors, Traditional Chinese Medicine, Naturopathic/Holistic, Acupuncture, Health & Medical, Nutritionists

=== Answer ===

You can go to Abby Rappoport, who offers services in Traditional Chinese Medicine, Naturopathic/Holistic approaches, Acupuncture, and Nutrition.

=== finished ===
```

# UI Experience

If you prefer a more "out of the box" or UI experience, you can also run a web application wrapper:

```bash
python app.py
```

This will give you options to run the same scripts (both data-loading and RAG) from a single web app.

![Loading data](/images/screenshotLoad.png "Loading dataset")

![Loading data](/images/screenshotRag.png "Performing RAG")

# Troubleshooting

Some things to check if you run into issues:

- **No results returned**: Verify data exists in the bucket/scope/collection and that filters (sender, radius, etc.) aren’t too restrictive.
- **Vector index errors**: Make sure the index is online and the `dimension` matches the embedding model.
- **Dimension mismatch**: If you changed embedding models, re-load the data and recreate the index.
- **Hybrid queries not working**: Confirm the FTS index exists and try increasing or removing the geo filter to validate data.
- **Composite filters not applying**: Check that filter fields are present, indexed, and have the expected data types.
- **Poor RAG answers**: Increase `LIMIT`, confirm the same embedding model is used for data and prompts, and inspect the retrieved context.

If in doubt, run a simple non-vector query first to confirm the data looks right.

# What's next?

* Composite: Try similar prompts with different senders/receivers to see how the result differ.
* Hybrid (FTS): Try a radius search from a few blocks away (with the same prompt) and see how the result differ.
* Hyperscale: Adjust tuneables like nprobes to see how the result differs.