function setLoadCmd(value) {
    document.getElementById('loadCmd').value = value;
}

function setRagCmd(value) {
    document.getElementById('ragCmd').value = value;
}

async function runLoad() {
    return runCommand('loadCmd', 'outputLoad', 'preflightLoad');
}

async function runRag() {
    return runCommand('ragCmd', 'outputQuery', 'preflightRag', 'memoryBadge');
}

// Watch the streamed RAG output for the AI Data Plane markers and show a badge:
// a green "⚡ from memory" when a repeat short-circuits, else "full RAG" with the total time.
function updateMemoryBadge(badge, text) {
    if (!badge) return;
    const hit = text.match(/Served from memory in ([\d.]+)s/);
    if (hit) {
        badge.textContent = '⚡ from memory · ' + hit[1] + 's';
        badge.className = 'preflight-badge pass';
        badge.dataset.settled = '1';
        return;
    }
    if (badge.dataset.settled === '1') return;
    const tot = text.match(/total ([\d.]+)s/);
    if (tot) {
        badge.textContent = 'full RAG · ' + tot[1] + 's';
        badge.className = 'preflight-badge mem-full';
    }
}

// Watch the streamed output for the scripts' preflight markers (#2) and reflect them
// on a status badge so the pass/fail is visible without scrolling the output.
function updatePreflightBadge(badge, text) {
    if (!badge || badge.dataset.settled === '1') return;
    if (text.includes('Preflight OK')) {
        badge.textContent = '✓ Preflight passed';
        badge.className = 'preflight-badge pass';
        badge.dataset.settled = '1';
    } else if (text.includes('Embedding dimension mismatch')) {
        badge.textContent = '✗ Dimension mismatch';
        badge.className = 'preflight-badge fail';
        badge.dataset.settled = '1';
    } else if (text.includes('Preflight skipped')) {
        badge.textContent = 'Preflight skipped';
        badge.className = 'preflight-badge skip';
        badge.dataset.settled = '1';
    }
}

async function runCommand(cmdInputId, outputId, badgeId, memBadgeId) {
    const cmd = document.getElementById(cmdInputId).value;
    const output = document.getElementById(outputId);
    const badge = badgeId ? document.getElementById(badgeId) : null;
    const memBadge = memBadgeId ? document.getElementById(memBadgeId) : null;

    output.value = '';
    output.scrollTop = 0;
    for (const b of [badge, memBadge]) {
        if (b) { b.textContent = ''; b.className = 'preflight-badge'; b.dataset.settled = '0'; }
    }

    const res = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        output.value += chunk;
        output.scrollTop = output.scrollHeight;
        updatePreflightBadge(badge, output.value);
        updateMemoryBadge(memBadge, output.value);
    }
}

// ---------------------------------------------------------------------------
// Global embedding controls (provider + model) — drive every load/RAG command.
// ---------------------------------------------------------------------------

function embeddingProvider() {
    const el = document.getElementById('globalProvider');
    return el ? el.value : 'local';
}

function embeddingModel() {
    const el = document.getElementById('globalModel');
    return el ? el.value.trim() : '';
}

// The configured bucket name, kept in sync with .env (COUCHBASE_BUCKET) so the generated
// commands target whatever bucket the user actually created — not a hardcoded default.
let demoBucket = 'vectorSearchDemo';

function bucketName() {
    return demoBucket || 'vectorSearchDemo';
}

// Agent-memory session base. The effective session is scoped per workflow (hyperscale/composite/
// hybrid) so memories from one workflow aren't recalled in another — the session updates as part of
// the workflow the user chooses. "New session" starts a fresh base for all workflows.
let sessionBase = 'default';

// Which RAG workflow tab is active (hyperscale / composite / hybrid), for display + scoping.
function activeWorkflow() {
    if (document.getElementById('tab4') && document.getElementById('tab4').checked) return 'composite';
    if (document.getElementById('tab6') && document.getElementById('tab6').checked) return 'hybrid';
    return 'hyperscale';  // tab5, default
}

// Effective per-workflow session id.
function sessionFor(workflow) {
    return `${sessionBase}-${workflow}`;
}

// Reflect the active workflow's session in the UI.
function updateSessionDisplay() {
    const el = document.getElementById('dpSession');
    if (el) el.textContent = sessionFor(activeWorkflow());
}

function newSession() {
    sessionBase = 'sess-' + Date.now().toString(36);
    updateSessionDisplay();
    const msg = document.getElementById('dpToggleMsg');
    if (msg) { msg.textContent = 'Started a fresh session for all workflows.'; msg.className = 'dp-msg'; }
}

// Memory recall relevance threshold (0-1) from the UI; blank = memory server default.
function memoryMinScore() {
    const el = document.getElementById('dpMinScore');
    return el ? el.value.trim() : '';
}

// Append the per-workflow agent-memory session and (optional) recall threshold to a RAG command.
function withRagSessionFlags(cmd, workflow) {
    cmd += ` --session ${sessionFor(workflow)}`;
    const ms = memoryMinScore();
    if (ms !== '') {
        cmd += ` --memory-min-score ${ms}`;
    }
    return cmd;
}

// Load the configured bucket from the backend on startup (and refresh it after a config save).
async function refreshBucket() {
    try {
        const res = await fetch('/config');
        const c = await res.json();
        if (c && c.bucket) demoBucket = c.bucket;
    } catch (e) {
        /* keep the default if the backend isn't reachable */
    }
}
document.addEventListener('DOMContentLoaded', refreshBucket);

// Map a base (local) collection name to the collection for the selected provider.
// OpenAI (1536-dim) data lives in a parallel "<base>_openai" collection.
function collectionFor(base) {
    return embeddingProvider() === 'openai' ? `${base}_openai` : base;
}

// Append --embedding-provider and (when set) --embedding-model to a command.
function withEmbeddingFlags(cmd) {
    cmd += ` --embedding-provider ${embeddingProvider()}`;
    const model = embeddingModel();
    if (model) {
        cmd += ` --embedding-model "${model}"`;
    }
    return cmd;
}

// Update the model placeholder to the provider's default when the provider changes.
function onProviderChange() {
    const el = document.getElementById('globalModel');
    if (el) {
        el.placeholder = embeddingProvider() === 'openai'
            ? 'text-embedding-3-small (1536-dim)'
            : 'all-MiniLM-L6-v2 (384-dim)';
    }
}

// ---------------------------------------------------------------------------
// Load builders
// ---------------------------------------------------------------------------

function setLoadCmdComposite() {
    const collection = collectionFor('emails');
    const cmd = `python load.py ` +
                `--data data/dataset.csv --text-fields subject message_body ` +
                `--bucket ${bucketName()} --scope _default --collection ${collection} ` +
                `--copy-fields subject sender receiver message_body ` +
                `--limit 5 --id-field sender timestamp`;
    setLoadCmd(withEmbeddingFlags(cmd));
}

function setLoadCmdHyperscale() {
    const collection = collectionFor('movies');
    const cmd = `python load.py ` +
                `--data data/wiki_movie_plots_deduped.csv --text-fields Plot ` +
                `--bucket ${bucketName()} --scope _default --collection ${collection} ` +
                `--copy-fields Title "Release Year" Director ` +
                `--limit 5 --id-field Title "Release Year"`;
    setLoadCmd(withEmbeddingFlags(cmd));
}

function setLoadCmdHybrid() {
    const collection = collectionFor('yelp');
    const cmd = `python load.py ` +
                `--data data/yelp_academic_dataset_business.json --text-fields categories ` +
                `--bucket ${bucketName()} --scope _default --collection ${collection} ` +
                `--copy-fields latitude longitude name ` +
                `--limit 5 --id-field business_id`;
    setLoadCmd(withEmbeddingFlags(cmd));
}

// ---------------------------------------------------------------------------
// RAG builders
// ---------------------------------------------------------------------------

function setRagCmdComposite() {
    const sender = document.getElementById('sender').value.trim();
    const receiver = document.getElementById('receiver').value.trim();
    const prompt = document.getElementById('prompt').value.trim();

    const collection = collectionFor('emails');

    let cmd = `python ragComposite.py ` +
              `--bucket ${bucketName()} ` +
              `--scope _default ` +
              `--collection ${collection}`;

    if (sender) {
        cmd += ` --sender "${sender}"`;
    }

    if (receiver) {
        cmd += ` --receiver "${receiver}"`;
    }

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    const nprobes = document.getElementById('compNprobes').value.trim();
    if (nprobes) {
        cmd += ` --nprobes ${nprobes}`;
    }

    setRagCmd(withEmbeddingFlags(withRagSessionFlags(cmd, "composite")));
}

function setRagCmdHyperscale() {
    const prompt = document.getElementById('hyperPrompt').value.trim();

    const collection = collectionFor('movies');

    let cmd = `python ragHyperscale.py ` +
              `--bucket ${bucketName()} ` +
              `--scope _default ` +
              `--collection ${collection}`;

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    const nprobes = document.getElementById('hyperNprobes').value.trim();
    if (nprobes) {
        cmd += ` --nprobes ${nprobes}`;
    }

    setRagCmd(withEmbeddingFlags(withRagSessionFlags(cmd, "hyperscale")));
}

function setRagCmdHybrid() {
    const prompt = document.getElementById('hybridPrompt').value.trim();

    const collection = collectionFor('yelp');
    // The FTS index is dimension-specific, so it swaps with the provider too.
    const indexName = embeddingProvider() === 'openai' ? 'ix-yelp-openai-vector' : 'ix-yelp-business-vector';

    const latitude = document.getElementById('hybridLatitude').value.trim();
    const longitude = document.getElementById('hybridLongitude').value.trim();
    const radius = document.getElementById('hybridRadius').value.trim();

    let cmd = `python ragHybrid.py ` +
              `--bucket ${bucketName()} ` +
              `--scope _default ` +
              `--collection ${collection} ` +
              `--index-name ${indexName}`;

    if (prompt) {
        cmd += ` --prompt "${prompt}"`;
    }

    if (latitude && longitude && radius) {
        cmd += ` --latitude ${latitude} --longitude ${longitude} --radius ${radius}`;
    }

    const numCandidates = document.getElementById('hybridNumCandidates').value.trim();
    if (numCandidates) {
        cmd += ` --num-candidates ${numCandidates}`;
    }

    setRagCmd(withEmbeddingFlags(withRagSessionFlags(cmd, "hybrid")));
}

// ---------------------------------------------------------------------------
// Configuration drawer: read/save .env and test connectivity via the backend.
// ---------------------------------------------------------------------------

function openConfig() {
    document.getElementById('configDrawer').classList.add('open');
    document.getElementById('configOverlay').classList.add('open');
    loadConfig();
}

function closeConfig() {
    document.getElementById('configDrawer').classList.remove('open');
    document.getElementById('configOverlay').classList.remove('open');
}

async function loadConfig() {
    const res = await fetch('/config');
    const c = await res.json();
    document.getElementById('cfgConnstr').value = c.connstr || '';
    document.getElementById('cfgUsername').value = c.username || '';
    document.getElementById('cfgPassword').value = c.password || '';   // masked
    document.getElementById('cfgBucket').value = c.bucket || '';
    if (c.bucket) demoBucket = c.bucket;   // keep generated commands in sync
    document.getElementById('cfgProvider').value = c.provider || 'local';
    document.getElementById('cfgModel').value = c.model || '';
    document.getElementById('cfgDimensions').value = c.dimensions || '';
    document.getElementById('cfgOpenaiKey').value = c.openai_key || '';  // masked
    document.getElementById('cfgOpenaiChat').value = c.openai_chat_model || '';
}

function collectConfig() {
    return {
        connstr: document.getElementById('cfgConnstr').value.trim(),
        username: document.getElementById('cfgUsername').value.trim(),
        password: document.getElementById('cfgPassword').value,          // '•' → keep existing
        bucket: document.getElementById('cfgBucket').value.trim(),
        provider: document.getElementById('cfgProvider').value,
        model: document.getElementById('cfgModel').value.trim(),
        dimensions: document.getElementById('cfgDimensions').value.trim(),
        openai_key: document.getElementById('cfgOpenaiKey').value,       // '•' → keep existing
        openai_chat_model: document.getElementById('cfgOpenaiChat').value.trim(),
    };
}

function setConfigStatus(msg, kind) {
    const el = document.getElementById('configStatus');
    el.textContent = msg;
    el.className = 'config-status' + (kind ? ' ' + kind : '');
}

async function saveConfig() {
    setConfigStatus('Saving…', '');
    try {
        const res = await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(collectConfig())
        });
        const data = await res.json();
        setConfigStatus(data.ok ? 'Saved to .env ✓' : 'Save failed', data.ok ? 'pass' : 'fail');
        if (data.ok) loadConfig();  // re-mask secrets from the saved file
    } catch (e) {
        setConfigStatus('Save failed: ' + e, 'fail');
    }
}

async function testConnection() {
    const btn = document.querySelector('.btn-test');
    setConfigStatus('Testing… (can take up to ~20s)', '');
    const list = document.getElementById('preflightResults');
    list.innerHTML = '';
    if (btn) btn.disabled = true;

    // Abort the request if it runs long, so we show a clear message instead of a raw
    // "Failed to fetch" network error.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45000);

    try {
        const res = await fetch('/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(collectConfig()),
            signal: controller.signal
        });
        if (!res.ok) {
            setConfigStatus('Test failed: server returned ' + res.status, 'fail');
            return;
        }
        const data = await res.json();
        for (const r of data.results) {
            const li = document.createElement('li');
            li.className = r.ok ? 'ok' : 'bad';
            li.textContent = (r.ok ? '✓ ' : '✗ ') + r.name + ': ' + r.detail;
            list.appendChild(li);
        }
        setConfigStatus(
            data.ok ? 'All checks passed — ready to load data.' : 'Some checks failed.',
            data.ok ? 'pass' : 'fail'
        );
    } catch (e) {
        const msg = (e && e.name === 'AbortError')
            ? 'Test timed out — check the connection string and that your IP is allowed in Capella (Step 0.5).'
            : 'Could not reach the server. Is app.py still running?';
        setConfigStatus(msg, 'fail');
    } finally {
        clearTimeout(timer);
        if (btn) btn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// AI Data Plane: diagram-tab status overlay + global RAG toggle.
// ---------------------------------------------------------------------------

async function refreshDataplane() {
    try {
        const res = await fetch('/dataplane/status');
        const st = await res.json();
        const toggle = document.getElementById('dpToggle');
        if (toggle) toggle.checked = !!st.enabled;

        const am = (st.capabilities && st.capabilities.agent_memory) || {};
        const dot = document.getElementById('am-dot');
        const stat = document.getElementById('am-stat');
        if (dot) dot.className = 'dp-dot ' + (am.active ? 'on' : 'off');
        if (stat) {
            if (!am.active) {
                stat.textContent = 'inactive — start memory_server.py';
            } else if (am.used) {
                stat.textContent = `active · ${am.memory_added} stored · ${am.searches} recalls · ~${am.tokens_served_from_memory_est} tokens served`;
            } else {
                stat.textContent = 'active · not used yet';
            }
        }
    } catch (e) {
        const stat = document.getElementById('am-stat');
        if (stat) stat.textContent = 'status unavailable (is app.py running?)';
    }
}

async function toggleDataplane() {
    const toggle = document.getElementById('dpToggle');
    const msg = document.getElementById('dpToggleMsg');
    try {
        const res = await fetch('/dataplane/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: toggle.checked })
        });
        const data = await res.json();
        if (msg) {
            msg.textContent = data.enabled ? 'RAG chats will use agent memory.' : 'RAG chats use plain RAG.';
            msg.className = 'dp-msg ' + (data.enabled ? 'on' : '');
        }
        refreshDataplane();
    } catch (e) {
        if (msg) { msg.textContent = 'toggle failed'; msg.className = 'dp-msg fail'; }
    }
}

// Poll status while the AI Services tab is selected (lightweight; only when visible).
setInterval(() => {
    const tab = document.getElementById('tab7');
    if (tab && tab.checked) refreshDataplane();
}, 5000);
document.addEventListener('DOMContentLoaded', () => {
    const tab = document.getElementById('tab7');
    if (tab) tab.addEventListener('change', () => { if (tab.checked) refreshDataplane(); });
    // Update the displayed session when the user switches RAG workflow (tabs 4/5/6).
    for (const id of ['tab4', 'tab5', 'tab6']) {
        const t = document.getElementById(id);
        if (t) t.addEventListener('change', updateSessionDisplay);
    }
    updateSessionDisplay();
    refreshDataplane();
});
