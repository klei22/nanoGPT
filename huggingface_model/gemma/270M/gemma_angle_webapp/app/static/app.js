const state = {
  tokenA: null,
  tokenB: null,
  anchor: null,
};

const pickerPrefixes = ['tokenA', 'tokenB', 'anchor'];

const cache = {
  tokenA: [],
  tokenB: [],
  anchor: [],
};

const searchRequestIds = {
  tokenA: 0,
  tokenB: 0,
  anchor: 0,
};

const lastSearchedQuery = {
  tokenA: null,
  tokenB: null,
  anchor: null,
};


function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatErrorDetail(detail) {
  if (!detail) return 'Request failed';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') return item;
        const location = Array.isArray(item.loc) ? item.loc.join('.') : item.loc;
        const message = item.msg || JSON.stringify(item);
        return location ? `${location}: ${message}` : message;
      })
      .join('; ');
  }
  try {
    return JSON.stringify(detail);
  } catch (_) {
    return String(detail);
  }
}

function tokenLabel(token) {
  if (!token) return 'No token selected.';
  const raw = JSON.stringify(token.raw);
  const display = token.display && token.display !== token.raw ? ` — ${JSON.stringify(token.display)}` : '';
  return `${String(token.token_id).padStart(6, ' ')} | ${raw}${display}`;
}

function setOutput(elementId, html, muted = false) {
  const el = byId(elementId);
  el.classList.toggle('muted', muted);
  el.innerHTML = html;
}

function setSelectMessage(select, message) {
  select.innerHTML = '';
  const option = document.createElement('option');
  option.textContent = message;
  option.value = '';
  select.appendChild(option);
}

function setBadge(text, className = null) {
  const badge = byId('statusBadge');
  badge.classList.remove('ready', 'error');
  if (className) badge.classList.add(className);
  badge.textContent = text;
}

function setPickerSelection(prefix, key, token, message = null) {
  state[key] = token;
  const idInput = byId(`${prefix}Id`);
  const selected = byId(`${prefix}Selected`);

  if (token) {
    idInput.value = String(token.token_id);
    selected.textContent = `Selected: ${tokenLabel(token)}`;
  } else {
    idInput.value = '';
    selected.textContent = message || 'No token selected.';
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = formatErrorDetail(data.detail || data);
    } catch (_) {
      // Keep response.statusText.
    }
    throw new Error(detail);
  }
  return response.json();
}

function updateStatusCard(status) {
  const loaded = Boolean(status.loaded);
  setBadge(loaded ? 'Model ready' : 'No model loaded', loaded ? 'ready' : null);
  byId('statusModel').textContent = status.model_name;
  byId('statusDevice').textContent = `${status.effective_device} (requested ${status.requested_device})`;
  byId('statusVocab').textContent = loaded ? status.vocab_size.toLocaleString() : '—';
  byId('statusHidden').textContent = loaded ? status.hidden_dim.toLocaleString() : '—';
  byId('modelNameInput').value = status.model_name;
  byId('modelDeviceInput').value = status.requested_device;
}


function formatBytes(bytes) {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const precision = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

async function loadAvailableModels() {
  const select = byId('localModelSelect');
  const refreshButton = byId('refreshLocalModelsButton');
  const currentInput = byId('modelNameInput').value.trim();

  select.disabled = true;
  refreshButton.disabled = true;
  setSelectMessage(select, 'Scanning local Hugging Face cache…');

  try {
    const data = await fetchJson('/api/models/available');
    select.innerHTML = '';

    if (!data.models || data.models.length === 0) {
      setSelectMessage(select, 'No locally cached models found');
      return;
    }

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = `Select a cached model… (${data.models.length})`;
    select.appendChild(placeholder);

    for (const model of data.models) {
      const option = document.createElement('option');
      option.value = model.model_name;
      const size = formatBytes(model.size_bytes);
      option.textContent = size ? `${model.model_name} — ${size}` : model.model_name;
      if (model.model_name === currentInput) option.selected = true;
      select.appendChild(option);
    }
  } catch (error) {
    setSelectMessage(select, `Cache scan failed: ${error.message}`);
  } finally {
    select.disabled = false;
    refreshButton.disabled = false;
  }
}

async function loadStatus() {
  setBadge('Checking status…');
  try {
    const status = await fetchJson('/api/status');
    updateStatusCard(status);
    byId('modelLoadMessage').textContent = status.loaded
      ? 'Active model loaded.'
      : 'No model loaded yet. Enter a Hugging Face model ID, choose a cached model, then click Load model.';
  } catch (error) {
    setBadge('Status failed', 'error');
    byId('modelLoadMessage').textContent = `Status failed: ${error.message}`;
  }
}

function resetPicker(prefix, key) {
  state[key] = null;
  cache[prefix] = [];
  searchRequestIds[prefix] += 1;
  lastSearchedQuery[prefix] = null;

  const select = byId(`${prefix}Results`);
  setSelectMessage(select, 'Click Search to load matches');
  select.disabled = false;
  select.classList.add('stale');

  const searchButton = byId(`${prefix}Search`);
  searchButton.disabled = false;
  searchButton.textContent = 'Search';
  searchButton.classList.add('needs-search');

  byId(`${prefix}Id`).value = '';
  byId(`${prefix}Selected`).textContent = prefix === 'anchor' ? 'No anchor selected.' : 'No token selected.';
}

function resetAllSelectionsAfterModelChange() {
  resetPicker('tokenA', 'tokenA');
  resetPicker('tokenB', 'tokenB');
  resetPicker('anchor', 'anchor');

  setOutput('angleOutput', 'Choose two tokens, then compute their angle.', true);
  setOutput('neighborhoodOutput', 'Choose an anchor token, then compute its closest tokens.', true);
  byId('neighborhoodTable').classList.add('hidden');
  byId('downloadCsv').classList.add('hidden');
  resetPairwiseBinsOutput();
}

async function loadRequestedModel() {
  const modelName = byId('modelNameInput').value.trim();
  const device = byId('modelDeviceInput').value.trim() || 'cpu';
  const allowDownload = byId('allowDownloadInput').checked;
  const button = byId('loadModelButton');
  const message = byId('modelLoadMessage');

  if (!modelName) {
    message.textContent = 'Enter a Hugging Face model ID before loading.';
    setBadge('Model ID required', 'error');
    return;
  }

  button.disabled = true;
  button.textContent = 'Loading…';
  setBadge('Loading model…');
  message.textContent = allowDownload
    ? `Loading ${modelName} on ${device}; downloading missing files if needed…`
    : `Loading ${modelName} on ${device} from the local cache only…`;

  try {
    const status = await fetchJson('/api/model/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName, device, force_reload: false, allow_download: allowDownload }),
    });
    updateStatusCard(status);
    resetAllSelectionsAfterModelChange();
    message.textContent = `Loaded ${status.model_name}.`;
    loadAvailableModels();
    setOutput('angleOutput', 'Choose two tokens, then compute their angle.', true);
    setOutput('neighborhoodOutput', 'Choose an anchor token, then compute its closest tokens.', true);
  } catch (error) {
    setBadge('Model load failed', 'error');
    message.textContent = `Model load failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = 'Load model';
  }
}

async function search(prefix, key) {
  const requestId = ++searchRequestIds[prefix];
  const query = byId(`${prefix}Query`).value;
  const select = byId(`${prefix}Results`);
  const searchButton = byId(`${prefix}Search`);
  select.disabled = true;
  searchButton.disabled = true;
  searchButton.textContent = 'Searching…';
  setSelectMessage(select, 'Searching…');

  try {
    const params = new URLSearchParams({ q: query });
    const data = await fetchJson(`/api/tokens/search?${params.toString()}`);

    // Ignore stale responses if another explicit search/model load started while this request was in flight.
    if (requestId !== searchRequestIds[prefix]) return;

    lastSearchedQuery[prefix] = query;
    searchButton.classList.remove('needs-search');
    searchButton.textContent = 'Search';
    select.classList.remove('stale');

    cache[prefix] = data.results;
    select.innerHTML = '';

    if (data.results.length === 0) {
      setSelectMessage(select, 'No matches');
      setPickerSelection(prefix, key, null, 'No matching tokens found.');
      markSearchStale(prefix, key);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const token of data.results) {
      const option = document.createElement('option');
      option.value = String(token.token_id);
      option.textContent = tokenLabel(token);
      fragment.appendChild(option);
    }
    select.appendChild(fragment);

    // Match Streamlit's selectbox behavior: when matches exist, one token is
    // selected immediately. Preserve the previous selection if it is still in
    // the visible result set; otherwise choose the first match.
    const previousId = state[key]?.token_id;
    const selected = data.results.find((token) => token.token_id === previousId) || data.results[0];
    select.value = String(selected.token_id);
    setPickerSelection(prefix, key, selected);
    markSearchStale(prefix, key);
  } catch (error) {
    if (requestId !== searchRequestIds[prefix]) return;
    cache[prefix] = [];
    setSelectMessage(select, `Error: ${error.message}`);
    setPickerSelection(prefix, key, null, `Search failed: ${error.message}`);
  } finally {
    if (requestId === searchRequestIds[prefix]) {
      select.disabled = false;
      searchButton.disabled = false;
      searchButton.textContent = searchButton.classList.contains('needs-search') ? 'Search updated query' : 'Search';
    }
  }
}

async function selectFromId(prefix, key) {
  const rawValue = byId(`${prefix}Id`).value;
  const tokenId = Number(rawValue);
  if (!Number.isInteger(tokenId) || tokenId < 0) {
    setPickerSelection(prefix, key, null, 'Enter a non-negative integer token ID.');
    return;
  }

  try {
    const token = await fetchJson(`/api/tokens/id/${tokenId}`);
    setPickerSelection(prefix, key, token);
  } catch (error) {
    setPickerSelection(prefix, key, null, `Token lookup failed: ${error.message}`);
  }
}

function selectFromDropdown(prefix, key) {
  const select = byId(`${prefix}Results`);
  const selectedId = Number(select.value);
  const token = cache[prefix].find((candidate) => candidate.token_id === selectedId);
  if (!token) return;
  setPickerSelection(prefix, key, token);
}

function markSearchStale(prefix, key) {
  // Query edits should not issue network requests. They only mark the picker
  // as needing an explicit Search click.
  const searchButton = byId(`${prefix}Search`);
  const select = byId(`${prefix}Results`);
  const query = byId(`${prefix}Query`).value;
  searchButton.disabled = false;

  if (lastSearchedQuery[prefix] === null) {
    searchButton.classList.add('needs-search');
    searchButton.textContent = 'Search';
    setSelectMessage(select, 'Click Search to load matches');
    select.classList.add('stale');
    return;
  }

  if (query !== lastSearchedQuery[prefix]) {
    searchButton.classList.add('needs-search');
    searchButton.textContent = 'Search updated query';
    select.classList.add('stale');
  } else {
    searchButton.classList.remove('needs-search');
    searchButton.textContent = 'Search';
    select.classList.remove('stale');
  }
}

function setupPicker(prefix, key) {
  const queryInput = byId(`${prefix}Query`);
  const searchButton = byId(`${prefix}Search`);
  const results = byId(`${prefix}Results`);

  setSelectMessage(results, 'Click Search to load matches');
  results.classList.add('stale');
  searchButton.classList.add('needs-search');

  queryInput.addEventListener('input', () => markSearchStale(prefix, key));
  queryInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      search(prefix, key);
    }
  });
  searchButton.addEventListener('click', () => search(prefix, key));
  byId(`${prefix}Results`).addEventListener('change', () => selectFromDropdown(prefix, key));
  byId(`${prefix}UseId`).addEventListener('click', () => selectFromId(prefix, key));
}

async function computeAngle() {
  if (!state.tokenA || !state.tokenB) {
    setOutput('angleOutput', 'Select Token A and Token B first.', true);
    return;
  }

  setOutput('angleOutput', 'Computing angle…', true);
  try {
    const data = await fetchJson(`/api/angle?token_a=${state.tokenA.token_id}&token_b=${state.tokenB.token_id}`);
    setOutput('angleOutput', `
      <div class="metric"><span>Angle</span><strong>${data.angle_deg.toFixed(6)}°</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Token A</th><td>${escapeHtml(tokenLabel({ token_id: data.token_a_id, raw: data.token_a_raw, display: data.token_a_display }))}</td></tr>
          <tr><th>Token A vector length</th><td>${data.token_a_magnitude.toFixed(6)}</td></tr>
          <tr><th>Token B</th><td>${escapeHtml(tokenLabel({ token_id: data.token_b_id, raw: data.token_b_raw, display: data.token_b_display }))}</td></tr>
          <tr><th>Token B vector length</th><td>${data.token_b_magnitude.toFixed(6)}</td></tr>
        </tbody>
      </table>
    `);
  } catch (error) {
    setOutput('angleOutput', `<strong>Angle failed:</strong> ${escapeHtml(error.message)}`);
  }
}

async function computeNeighborhood() {
  if (!state.anchor) {
    setOutput('neighborhoodOutput', 'Select an anchor token first.', true);
    return;
  }

  const limit = Number(byId('neighborhoodLimit').value || 500);
  setOutput('neighborhoodOutput', 'Computing neighborhood…', true);
  byId('neighborhoodTable').classList.add('hidden');
  byId('downloadCsv').classList.add('hidden');

  try {
    const data = await fetchJson(`/api/neighborhood?anchor_id=${state.anchor.token_id}&limit=${limit}`);
    setOutput('neighborhoodOutput', `
      <div class="metric"><span>Anchor vector length</span><strong>${data.anchor_magnitude.toFixed(6)}</strong></div>
      <p>Anchor: ${escapeHtml(tokenLabel({ token_id: data.anchor_id, raw: data.anchor_raw, display: data.anchor_display }))}</p>
    `);

    const tbody = byId('neighborhoodTable').querySelector('tbody');
    tbody.innerHTML = '';
    for (const row of data.rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${row.rank}</td>
        <td>${row.token_id}</td>
        <td><code>${escapeHtml(row.token_raw)}</code></td>
        <td><code>${escapeHtml(row.token_display)}</code></td>
        <td>${row.angle_deg.toFixed(6)}</td>
        <td>${row.magnitude.toFixed(6)}</td>
      `;
      tbody.appendChild(tr);
    }

    const table = byId('neighborhoodTable');
    table.classList.remove('hidden');

    const download = byId('downloadCsv');
    download.href = `/api/neighborhood.csv?anchor_id=${state.anchor.token_id}`;
    download.classList.remove('hidden');
  } catch (error) {
    setOutput('neighborhoodOutput', `<strong>Neighborhood failed:</strong> ${escapeHtml(error.message)}`);
  }
}


function resetPairwiseBinsOutput() {
  const output = byId('pairwiseBinsOutput');
  if (!output) return;
  setOutput('pairwiseBinsOutput', 'Load a model, then compute the global pairwise angle-bin distribution. A CUDA device is used automatically when available.', true);
  byId('pairwisePlotCard').classList.add('hidden');
  byId('pairwiseBinsTable').classList.add('hidden');
  byId('pairwiseBinsTable').querySelector('tbody').innerHTML = '';
}

function formatCount(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toLocaleString() : String(value);
}

function drawPairwiseRankPlot(bins) {
  const canvas = byId('pairwiseAnglePlot');
  const card = byId('pairwisePlotCard');
  const ctx = canvas.getContext('2d');
  const style = getComputedStyle(document.documentElement);
  const textColor = style.getPropertyValue('--text').trim() || '#e5e7eb';
  const mutedColor = style.getPropertyValue('--muted').trim() || '#9ca3af';
  const borderColor = style.getPropertyValue('--border').trim() || '#374151';
  const accentColor = style.getPropertyValue('--accent-strong').trim() || '#60a5fa';

  const cssWidth = Math.max(720, card.clientWidth || 1000);
  const cssHeight = 420;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const margin = { top: 28, right: 26, bottom: 58, left: 76 };
  const width = cssWidth - margin.left - margin.right;
  const height = cssHeight - margin.top - margin.bottom;
  const counts = bins.map((bin) => Number(bin.count || 0));
  const logs = counts.map((count) => Math.log10(Math.max(1, count)));
  const maxLog = Math.max(1, ...logs);
  const minLog = 0;
  const rankCount = Math.max(1, bins.length);

  function xForRank(rank) {
    if (rankCount === 1) return margin.left + width / 2;
    return margin.left + ((rank - 1) / (rankCount - 1)) * width;
  }

  function yForLog(logValue) {
    return margin.top + ((maxLog - logValue) / (maxLog - minLog)) * height;
  }

  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + height);
  ctx.lineTo(margin.left + width, margin.top + height);
  ctx.stroke();

  ctx.fillStyle = mutedColor;
  ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = Math.max(2, Math.ceil(maxLog));
  for (let tick = 0; tick <= yTicks; tick += 1) {
    const logValue = (tick / yTicks) * maxLog;
    const y = yForLog(logValue);
    ctx.strokeStyle = borderColor;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + width, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const countLabel = Math.round(10 ** logValue).toLocaleString();
    ctx.fillText(countLabel, margin.left - 10, y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xTickStep = Math.max(1, Math.ceil(rankCount / 6));
  for (let rank = 1; rank <= rankCount; rank += xTickStep) {
    const x = xForRank(rank);
    ctx.fillText(String(rank), x, margin.top + height + 10);
  }
  if (rankCount > 1 && (rankCount - 1) % xTickStep !== 0) {
    ctx.fillText(String(rankCount), xForRank(rankCount), margin.top + height + 10);
  }

  ctx.save();
  ctx.translate(18, margin.top + height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.fillText('Pair count, log scale', 0, 0);
  ctx.restore();

  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.fillText('Angle-bin rank', margin.left + width / 2, cssHeight - 22);

  ctx.fillStyle = accentColor;
  for (const bin of bins) {
    const x = xForRank(Number(bin.rank));
    const y = yForLog(Math.log10(Math.max(1, Number(bin.count || 0))));
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderPairwiseBinsTable(bins) {
  const table = byId('pairwiseBinsTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';
  for (const bin of bins) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${bin.rank}</td>
      <td>${escapeHtml(bin.label)}</td>
      <td>${bin.bin_index}</td>
      <td>${formatCount(bin.count)}</td>
    `;
    tbody.appendChild(tr);
  }
  table.classList.remove('hidden');
}

async function computePairwiseAngleBins() {
  const button = byId('pairwiseBinsButton');
  const blockSize = Number(byId('pairwiseBlockSize').value || 2048);
  const computeDevice = byId('pairwiseComputeDevice').value.trim() || 'auto';
  const includeSelf = byId('pairwiseIncludeSelf').checked;

  byId('pairwisePlotCard').classList.add('hidden');
  byId('pairwiseBinsTable').classList.add('hidden');
  setOutput('pairwiseBinsOutput', 'Computing all-pairs angle bins block by block… The full pairwise matrix is not cached or written to disk.', true);
  button.disabled = true;
  button.textContent = 'Computing…';

  try {
    const params = new URLSearchParams({
      block_size: String(blockSize),
      compute_device: computeDevice,
      include_self: String(includeSelf),
    });
    const data = await fetchJson(`/api/pairwise-angle-bins?${params.toString()}`);
    setOutput('pairwiseBinsOutput', `
      <div class="metric"><span>Total pairs binned</span><strong>${formatCount(data.total_pairs)}</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Model</th><td>${escapeHtml(data.model_name)}</td></tr>
          <tr><th>Vocab × hidden</th><td>${formatCount(data.vocab_size)} × ${formatCount(data.hidden_dim)}</td></tr>
          <tr><th>Angle definition</th><td>Acute angle, abs(cosine), 0°–90°</td></tr>
          <tr><th>Bin width</th><td>${data.bin_degrees}°</td></tr>
          <tr><th>Compute device</th><td>${escapeHtml(data.compute_device)}</td></tr>
          <tr><th>Block size</th><td>${formatCount(data.block_size)}</td></tr>
          <tr><th>Self-pairs</th><td>${data.include_self ? 'included' : 'excluded'}</td></tr>
          <tr><th>Elapsed</th><td>${Number(data.elapsed_seconds).toFixed(3)} seconds</td></tr>
        </tbody>
      </table>
    `);
    byId('pairwisePlotCard').classList.remove('hidden');
    drawPairwiseRankPlot(data.bins);
    renderPairwiseBinsTable(data.bins);
  } catch (error) {
    setOutput('pairwiseBinsOutput', `<strong>Pairwise binning failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Compute pairwise bins';
  }
}

function setupModelControls() {
  const loadButton = byId('loadModelButton');
  const modelInput = byId('modelNameInput');
  const deviceInput = byId('modelDeviceInput');
  const localModelSelect = byId('localModelSelect');
  const refreshLocalModelsButton = byId('refreshLocalModelsButton');

  loadButton.addEventListener('click', loadRequestedModel);
  localModelSelect.addEventListener('change', () => {
    if (localModelSelect.value) modelInput.value = localModelSelect.value;
  });
  refreshLocalModelsButton.addEventListener('click', loadAvailableModels);
  for (const input of [modelInput, deviceInput]) {
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        loadRequestedModel();
      }
    });
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setupModelControls();
  loadStatus();
  loadAvailableModels();
  setupPicker('tokenA', 'tokenA');
  setupPicker('tokenB', 'tokenB');
  setupPicker('anchor', 'anchor');
  byId('angleButton').addEventListener('click', computeAngle);
  byId('neighborhoodButton').addEventListener('click', computeNeighborhood);
  byId('pairwiseBinsButton').addEventListener('click', computePairwiseAngleBins);
});
