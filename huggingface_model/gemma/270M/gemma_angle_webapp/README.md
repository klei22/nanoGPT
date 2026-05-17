# LM-Head Angle Explorer

A regular Python webapp refactor of the Streamlit token-angle explorer. The app uses FastAPI for HTTP endpoints and a small vanilla JavaScript front end.

## What changed

- No Streamlit rerun loop. The browser calls JSON endpoints only when the user clicks Search, presses Enter in a search box, or computes a result.
- The active model is chosen from the browser: type any AutoModelForCausalLM-compatible Hugging Face repo ID into the **Hugging Face model** field, or pick one from the locally cached model dropdown, and click **Load model**.
- **Download if missing** is enabled by default. Leave it checked to allow `from_pretrained(...)` to download uncached model/tokenizer files from the Hugging Face Hub, or uncheck it for cache-only loading.
- The local-model dropdown is populated by passively scanning the Hugging Face cache; it does not download or load model tensors.
- The app no longer loads the default model just because the page opened; `/api/status` is cheap, and model loading begins only after the explicit **Load model** action.
- Model assets are reused across requests until another model is loaded.
- Only the tokenizer, LM-head/output-embedding weight matrix, and vector magnitudes / L2 lengths are retained after model loading. The loader now prefers a safetensors-only path so Gemma 3 can load even when importing `Gemma3ForCausalLM` is broken by optional compiled-extension version conflicts.
- CUDA device strings such as `cuda:0` are validated; if CUDA is requested but unavailable, the app falls back to CPU.
- Neighborhood table rendering is limited in the browser, while the full sorted CSV can be downloaded from a streaming endpoint.
- A new all-pairs angle-distribution section computes unique unordered token pairs block by block, immediately bins them into 5° acute-angle buckets from 0° to 90°, and plots the sorted bin counts on a log-scaled y axis without caching the full pairwise matrix.
- Token search route ordering is guarded so `/api/tokens/search` is not mistaken for a token ID route.
- Search dropdowns now auto-select a result after an explicit Search click, matching Streamlit selectbox behavior without searching on every keystroke.
- Token search is explicit and literal: click Search or press Enter, and the backend returns every case-insensitive substring match. Byte-fallback tokens such as `<0xF9>` also expose plain aliases like `0xF9` and `F9` for literal search.

## Project layout

```text
app/
  main.py              FastAPI routes and web entry point
  model_service.py     model loading, token search, angle math, CSV generation, all-pairs binning
  schemas.py           response models
  templates/index.html browser UI
  static/app.js        front-end behavior
  static/styles.css    styling
requirements.txt
.env.example
```

## Magnitude definition

The displayed `magnitude` is the Euclidean length of the token's LM-head/output-embedding row vector:

```text
magnitude[token_id] = sqrt(sum_j output_weight[token_id, j]^2)
```

In code this is computed explicitly as `torch.linalg.vector_norm(weight, ord=2, dim=1)`, where `weight` has shape `[vocab_size, hidden_dim]`.

## All-pairs angle distribution

The bottom section computes a global distribution for the currently loaded model. It uses the LM-head/output-embedding rows, normalizes row blocks on the selected compute device, multiplies one row block by one column block, and immediately reduces the resulting dot products into 5-degree bins. The full `vocab_size × vocab_size` matrix is never materialized, saved, or cached.

By default, it counts unique unordered pairs `i < j` and excludes self-pairs. Checking **Include self-pairs** adds `i == j`, which all land in the `0–5°` bin. The angle is an acute angle computed from `abs(cosine)`, so bins cover `0°–90°`; a signed vector angle would require `0°–180°` bins instead.

The **Compute device** field defaults to `auto`. If the active LM-head tensor is already on CUDA, that CUDA device is used. If the model was loaded on CPU but CUDA is available, the app streams blocks to `cuda:0` so the expensive block matrix multiplications are GPU accelerated while keeping VRAM bounded. The **CUDA/CPU block size** controls the temporary product block size; reduce it if CUDA reports out-of-memory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If the model requires Hugging Face authentication in your environment, run:

```bash
huggingface-cli login
```

## Run

From the project root:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

You can also run:

```bash
python -m app.main
```

## Loading another model

Use the model loader in the upper-right card. Example values:

```text
google/gemma-3-270m
Qwen/Qwen3.5-0.8B-Base
meta-llama/Llama-3.2-1B
```

Click **Load model**. With **Download if missing** checked, the app allows Hugging Face Hub/Transformers to download missing files into the normal local cache before loading. With it unchecked, the app passes `local_files_only=True` and will fail clearly if the requested files are not already cached. After a successful load, all token selections are cleared because token IDs and vocabulary text belong to the newly active tokenizer/model. Search, angle, and neighborhood endpoints return a clear "No model is loaded" error until this button has completed successfully.

The default load strategy is `auto`: first load only the output vector matrix from safetensors (`lm_head.weight`, or a tied embedding such as `model.embed_tokens.weight`), then fall back to `AutoModelForCausalLM.from_pretrained(...)` only if a usable safetensors matrix is unavailable. This is intentionally more robust for this app because no forward pass or generation is needed.

The **Available locally** dropdown is populated from cached Hugging Face model repos such as `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base`. Use **Refresh** after downloading a new model if it does not appear immediately.

## Configuration

Set environment variables before starting the app:

```bash
export MODEL_NAME=google/gemma-3-270m
export DEVICE=cpu        # cpu, cuda:0, or auto
export PORT=8000
export PAIRWISE_BLOCK_SIZE=2048  # default block size for all-pairs binning
export MODEL_LOAD_STRATEGY=auto    # auto, weight_only, safetensors, or full_model
```

Optional loader settings:

```bash
export ATTN_IMPLEMENTATION=eager  # default; unset or empty to omit this kwarg
export TRUST_REMOTE_CODE=false    # set true only for repos you trust
```

## Troubleshooting Gemma 3 model-load errors

If you see an error like:

```text
Could not import module 'Gemma3ForCausalLM'
Skipping import of cpp extensions due to incompatible torch version
```

that usually means the full Transformers model import path is tripping over an environment/version mismatch, often from an optional compiled package such as `torchao`/`fbgemm_gpu`. This app does not need those compiled generation/inference extensions. Leave `MODEL_LOAD_STRATEGY=auto` or set:

```bash
export MODEL_LOAD_STRATEGY=weight_only
```

Then restart the server and click **Load model** again. The weight-only path reads the output vectors directly from safetensors and avoids importing `Gemma3ForCausalLM`. If the selected repo has no safetensors checkpoint or no recognizable output/embedding matrix, use `MODEL_LOAD_STRATEGY=full_model` and fix the underlying PyTorch/Transformers/optional-extension environment.

## API endpoints

- `GET /` — browser UI
- `GET /api/status` — active model name, device, vocab size, hidden dimension
- `GET /api/models/available` — list Hugging Face model repos already present in the local cache
- `POST /api/model/load` — load/switch the active model. JSON body: `{"model_name":"Qwen/Qwen3.5-0.8B-Base", "device":"cpu", "allow_download":true}`; `device` is optional and `allow_download` defaults to true
- `GET /api/tokens/search?q=...` — case-insensitive literal token search; returns all matching tokens with no result cap
- `GET /api/tokens/id/{token_id}` — explicit token lookup by ID, used by the front end
- `GET /api/tokens/{token_id}` — backward-compatible token lookup by ID
- `GET /api/angle?token_a=0&token_b=1` — pairwise angle and vector magnitudes / L2 lengths
- `GET /api/neighborhood?anchor_id=0&limit=500` — nearest tokens by LM-head angle
- `GET /api/neighborhood.csv?anchor_id=0` — full sorted neighborhood CSV
- `GET /api/pairwise-angle-bins?block_size=2048&compute_device=auto&include_self=false` — blockwise all-pairs acute-angle histogram, returned as ranked 5° bins for plotting
