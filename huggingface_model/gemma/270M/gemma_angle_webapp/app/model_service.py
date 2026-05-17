from __future__ import annotations

import csv
import gc
import io
import os
import threading
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-3-270m")
DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "eager")
DEFAULT_TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").casefold() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True)
class TokenInfo:
    token_id: int
    raw: str
    display: str
    normalized: str
    present_in_tokenizer: bool = True


@dataclass(frozen=True)
class LocalModelInfo:
    model_name: str
    cache_path: str | None = None
    size_bytes: int | None = None
    last_modified: float | None = None


@dataclass(frozen=True)
class ModelAssets:
    model_name: str
    requested_device: str
    effective_device: str
    tokenizer: PreTrainedTokenizerBase
    token_infos: list[TokenInfo]
    weight: torch.Tensor
    magnitudes: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.weight.shape[1])

    def token(self, token_id: int) -> TokenInfo:
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"token_id must be in [0, {self.vocab_size - 1}], got {token_id}")
        return self.token_infos[token_id]


_ASSETS_LOCK = threading.RLock()
_ASSETS: ModelAssets | None = None


def _norm(text: str) -> str:
    return text.casefold().strip()


def _display_token(token: str) -> str:
    cleaned = token.replace("▁", " ")
    return cleaned.encode("utf-8", "replace").decode("utf-8")


_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _dedupe_texts(values: list[str]) -> tuple[str, ...]:
    """Return non-empty strings in first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _search_text_variants(raw: str, display: str) -> tuple[str, ...]:
    """Build literal-search variants for a token.

    Search is intentionally case-insensitive substring matching, not pattern matching.
    Besides the raw and display text, we include common token-content aliases so
    byte-fallback tokens like ``<0xF9>`` can still be found with plain ``F9``.
    """
    values: list[str] = []

    for text in (raw, display):
        if text is None:
            continue
        stripped = text.strip()
        values.extend([text, stripped])

        if stripped.startswith("▁"):
            values.append(stripped[1:])

        space_normalized = stripped.replace("▁", " ")
        values.extend([space_normalized, space_normalized.strip()])

        if stripped.startswith("<") and stripped.endswith(">"):
            inner = stripped[1:-1].strip()
            if inner and "<" not in inner and ">" not in inner:
                values.append(inner)
                if inner.casefold().startswith("0x") and len(inner) > 2:
                    hex_part = inner[2:]
                    if all(char in _HEX_DIGITS for char in hex_part):
                        values.append(hex_part)

    return _dedupe_texts(values)


def _normalized_search_text(raw: str, display: str) -> str:
    return " ".join(_norm(value) for value in _search_text_variants(raw, display))

def _build_token_infos(tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> list[TokenInfo]:
    """Return TokenInfo records indexed by token_id.

    The Streamlit version sorted vocab entries and then indexed into the sorted list.
    This version explicitly indexes by token_id so it keeps working if a tokenizer has
    sparse or out-of-order vocabulary IDs.
    """
    infos: list[TokenInfo | None] = [None] * vocab_size
    for tok, idx in tokenizer.get_vocab().items():
        if 0 <= idx < vocab_size:
            display = _display_token(tok)
            infos[idx] = TokenInfo(
                token_id=idx,
                raw=tok,
                display=display,
                normalized=_normalized_search_text(tok, display),
            )

    for idx, info in enumerate(infos):
        if info is None:
            placeholder = f"<missing-token-{idx}>"
            infos[idx] = TokenInfo(
                token_id=idx,
                raw=placeholder,
                display=placeholder,
                normalized=_normalized_search_text(placeholder, placeholder),
                present_in_tokenizer=False,
            )

    return [info for info in infos if info is not None]


def resolve_device(requested_device: str | None = None) -> str:
    requested = (requested_device or DEFAULT_DEVICE or "cpu").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    # Validate the device string early so bad env vars fail clearly.
    try:
        torch.device(requested)
    except Exception as exc:  # pragma: no cover - exact exception differs by torch version
        raise ValueError(f"Invalid torch device {requested!r}") from exc
    return requested


def _tokenizer_from_pretrained_kwargs(*, allow_download: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": not allow_download}
    if DEFAULT_TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = True
    return kwargs


def _model_from_pretrained_kwargs(*, allow_download: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = _tokenizer_from_pretrained_kwargs(allow_download=allow_download)
    if DEFAULT_ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = DEFAULT_ATTN_IMPLEMENTATION
    return kwargs


def _load_tokenizer(model_name: str, *, allow_download: bool = True) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        model_name,
        **_tokenizer_from_pretrained_kwargs(allow_download=allow_download),
    )


def _load_causal_lm(model_name: str, *, allow_download: bool = True):
    """Load a causal LM, falling back if a model rejects attn_implementation.

    Gemma works with the default eager-attention kwarg, but some other Hugging
    Face model classes either do not accept that kwarg or reject the requested
    implementation. Dropping only that kwarg makes the browser model picker more
    forgiving while preserving the original default behavior when possible.
    """
    kwargs = _model_from_pretrained_kwargs(allow_download=allow_download)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" not in kwargs:
            raise
        if isinstance(exc, ValueError) and "attn_implementation" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)


def _output_embedding_weight(model: Any) -> torch.Tensor:
    """Return the LM-head/output-embedding matrix for a causal LM.

    Most decoder-only Hugging Face models expose this as ``model.lm_head.weight``.
    ``get_output_embeddings()`` is the more general interface, so we prefer it and
    keep ``lm_head`` as a fallback.
    """
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        output_embeddings = get_output_embeddings()
        if output_embeddings is not None and getattr(output_embeddings, "weight", None) is not None:
            return output_embeddings.weight

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        return lm_head.weight

    raise ValueError(
        "Could not find an LM-head/output-embedding weight matrix on this model. "
        "Use an AutoModelForCausalLM-compatible Hugging Face repo."
    )


def _release_cached_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vector_magnitudes(weight: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean length of each token vector.

    The output embedding / LM-head weight matrix has shape
    ``[vocab_size, hidden_dim]``. Each row is one token's vector, so reducing over
    ``dim=1`` computes one scalar length per token:

        sqrt(sum_j weight[token_id, j] ** 2)

    This is the vector magnitude / L2 norm / Euclidean length.
    """
    return torch.linalg.vector_norm(weight, ord=2, dim=1)


def _target_model_and_device(
    model_name: str | None,
    requested_device: str | None,
) -> tuple[str, str, str]:
    """Resolve requested model/device, defaulting to the active model when present."""
    if model_name is None and requested_device is None and _ASSETS is not None:
        requested = _ASSETS.requested_device
        return _ASSETS.model_name, requested, _ASSETS.effective_device

    target_model = (model_name or DEFAULT_MODEL_NAME).strip()
    if not target_model:
        raise ValueError("Model name cannot be empty.")

    requested = (requested_device or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
    effective = resolve_device(requested)
    return target_model, requested, effective


def get_assets(
    model_name: str | None = None,
    requested_device: str | None = None,
    *,
    force_reload: bool = False,
    allow_download: bool = True,
) -> ModelAssets:
    """Return active model assets, loading or replacing them when needed.

    Calling ``get_assets()`` with no arguments returns the currently active model,
    or lazily loads the configured default model if nothing has been loaded yet.
    Calling it with ``model_name`` switches the active model to that Hugging Face
    repository ID when necessary.
    """
    global _ASSETS

    with _ASSETS_LOCK:
        target_model, requested, effective = _target_model_and_device(model_name, requested_device)

        if (
            not force_reload
            and _ASSETS is not None
            and _ASSETS.model_name == target_model
            and _ASSETS.requested_device == requested
            and _ASSETS.effective_device == effective
        ):
            return _ASSETS

        tokenizer = _load_tokenizer(target_model, allow_download=allow_download)
        model = _load_causal_lm(target_model, allow_download=allow_download)
        model.to(effective)
        model.eval()

        with torch.inference_mode():
            weight = _output_embedding_weight(model).detach().to(device=effective, dtype=torch.float32)
            magnitudes = vector_magnitudes(weight)

        token_infos = _build_token_infos(tokenizer, vocab_size=int(weight.shape[0]))

        # Keep only the tokenizer plus the tensors needed by the app.
        del model
        _release_cached_cuda_memory()

        _ASSETS = ModelAssets(
            model_name=target_model,
            requested_device=requested,
            effective_device=effective,
            tokenizer=tokenizer,
            token_infos=token_infos,
            weight=weight,
            magnitudes=magnitudes,
        )
        return _ASSETS


def get_current_assets() -> ModelAssets:
    """Return the already-loaded active model.

    The browser model picker is explicit: the server should not begin a large
    default-model download just because the page loaded or a status request ran.
    Endpoints that need tensors call this helper and return a clear error until
    the user clicks **Load model**.
    """
    with _ASSETS_LOCK:
        if _ASSETS is None:
            raise ValueError(
                "No model is loaded. Enter a Hugging Face model ID and click Load model first."
            )
        return _ASSETS


def get_model_status() -> dict[str, Any]:
    """Return model status without loading any model assets."""
    with _ASSETS_LOCK:
        if _ASSETS is not None:
            return {
                "loaded": True,
                "model_name": _ASSETS.model_name,
                "requested_device": _ASSETS.requested_device,
                "effective_device": _ASSETS.effective_device,
                "vocab_size": _ASSETS.vocab_size,
                "hidden_dim": _ASSETS.hidden_dim,
            }

    requested = (DEFAULT_DEVICE or "cpu").strip() or "cpu"
    try:
        effective = resolve_device(requested)
    except ValueError:
        # Keep the page usable even if DEVICE is misconfigured; the explicit
        # load request will return the detailed validation error.
        effective = requested

    return {
        "loaded": False,
        "model_name": DEFAULT_MODEL_NAME,
        "requested_device": requested,
        "effective_device": effective,
        "vocab_size": 0,
        "hidden_dim": 0,
    }


def load_active_model(
    model_name: str,
    requested_device: str | None = None,
    *,
    force_reload: bool = False,
    allow_download: bool = True,
) -> ModelAssets:
    """Load a Hugging Face causal LM and make it the active app model.

    When allow_download is true, transformers may download missing files from
    the Hub. When false, loading is cache-only via local_files_only=True.
    """
    return get_assets(
        model_name,
        requested_device or DEFAULT_DEVICE,
        force_reload=force_reload,
        allow_download=allow_download,
    )


def active_model_name() -> str:
    """Return the current model name, or the configured default before first load."""
    with _ASSETS_LOCK:
        return _ASSETS.model_name if _ASSETS is not None else DEFAULT_MODEL_NAME



def _coerce_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_hf_cache_dirs() -> list[Path]:
    """Return likely Hugging Face Hub cache directories without requiring hub imports."""
    candidates: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.getenv(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _record_local_model(
    records: dict[str, LocalModelInfo],
    *,
    model_name: str,
    cache_path: str | None = None,
    size_bytes: int | None = None,
    last_modified: float | None = None,
) -> None:
    if not model_name:
        return

    existing = records.get(model_name)
    if existing is None:
        records[model_name] = LocalModelInfo(
            model_name=model_name,
            cache_path=cache_path,
            size_bytes=size_bytes,
            last_modified=last_modified,
        )
        return

    # Prefer richer metadata when the same repo is seen from multiple scans.
    records[model_name] = LocalModelInfo(
        model_name=model_name,
        cache_path=existing.cache_path or cache_path,
        size_bytes=existing.size_bytes if existing.size_bytes is not None else size_bytes,
        last_modified=(
            max(x for x in [existing.last_modified, last_modified] if x is not None)
            if existing.last_modified is not None or last_modified is not None
            else None
        ),
    )


def _scan_cache_with_hub(records: dict[str, LocalModelInfo]) -> None:
    """Use huggingface_hub.scan_cache_dir when available."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return

    cache_dirs = _candidate_hf_cache_dirs()
    if not cache_dirs:
        cache_dirs = [None]  # type: ignore[list-item]

    for cache_dir in cache_dirs:
        try:
            if cache_dir is not None and not cache_dir.exists():
                continue
            cache_info = scan_cache_dir(cache_dir=cache_dir)
        except Exception:
            continue

        for repo in getattr(cache_info, "repos", []):
            repo_type = getattr(repo, "repo_type", "model")
            if repo_type not in (None, "model"):
                continue
            repo_id = getattr(repo, "repo_id", "") or ""
            repo_path = getattr(repo, "repo_path", None)
            size_on_disk = getattr(repo, "size_on_disk", None)
            last_modified = _coerce_timestamp(getattr(repo, "last_modified", None))
            _record_local_model(
                records,
                model_name=str(repo_id),
                cache_path=str(repo_path) if repo_path else None,
                size_bytes=int(size_on_disk) if size_on_disk is not None else None,
                last_modified=last_modified,
            )


def _scan_cache_manually(records: dict[str, LocalModelInfo]) -> None:
    """Fallback scanner for the standard HF cache layout: models--org--repo."""
    for cache_dir in _candidate_hf_cache_dirs():
        if not cache_dir.exists() or not cache_dir.is_dir():
            continue
        for repo_dir in cache_dir.glob("models--*"):
            if not repo_dir.is_dir():
                continue
            encoded = repo_dir.name[len("models--"):]
            model_name = encoded.replace("--", "/")
            try:
                modified = repo_dir.stat().st_mtime
            except OSError:
                modified = None
            _record_local_model(
                records,
                model_name=model_name,
                cache_path=str(repo_dir),
                size_bytes=None,
                last_modified=modified,
            )


def list_local_models() -> list[LocalModelInfo]:
    """Return Hugging Face model repos already present in the local cache.

    This is a passive cache scan. It does not download anything and it does not
    load model tensors. The standard Hub cache stores model repos as directories
    named like ``models--Qwen--Qwen3.5-0.8B-Base``; those are displayed as
    ``Qwen/Qwen3.5-0.8B-Base`` in the browser dropdown.
    """
    records: dict[str, LocalModelInfo] = {}
    _scan_cache_with_hub(records)
    _scan_cache_manually(records)
    return sorted(records.values(), key=lambda item: item.model_name.casefold())

def search_tokens(assets: ModelAssets, query: str) -> list[TokenInfo]:
    """Return every token matching a case-insensitive literal substring query.

    There is deliberately no result cap. A blank query returns the full
    vocabulary in token-id order.
    """
    q = _norm(query)
    if not q:
        return list(assets.token_infos)
    return [info for info in assets.token_infos if q in info.normalized]


@torch.inference_mode()
def angle_degrees(assets: ModelAssets, id_a: int, id_b: int) -> float:
    assets.token(id_a)
    assets.token(id_b)
    denom = (assets.magnitudes[id_a] * assets.magnitudes[id_b]).clamp_min(1e-12)
    cos_val = torch.dot(assets.weight[id_a], assets.weight[id_b]) / denom
    cos_val = torch.clamp(cos_val, -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos_val)).item())


@torch.inference_mode()
def _angles_for_anchor(assets: ModelAssets, anchor_id: int) -> torch.Tensor:
    assets.token(anchor_id)
    anchor = assets.weight[anchor_id]
    anchor_norm = assets.magnitudes[anchor_id].clamp_min(1e-12)
    denom = (assets.magnitudes * anchor_norm).clamp_min(1e-12)
    cos = (assets.weight @ anchor) / denom
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def _neighborhood_row(assets: ModelAssets, token_id: int, angle_deg: float, rank: int) -> dict[str, Any]:
    info = assets.token(token_id)
    return {
        "rank": rank,
        "token_id": token_id,
        "token_raw": info.raw,
        "token_display": info.display,
        "angle_deg": angle_deg,
        "magnitude": float(assets.magnitudes[token_id].item()),
    }


@torch.inference_mode()
def nearest_neighbors(
    assets: ModelAssets,
    anchor_id: int,
    limit: int,
    *,
    include_self: bool = True,
) -> list[dict[str, Any]]:
    angles = _angles_for_anchor(assets, anchor_id)
    k = min(assets.vocab_size, max(1, limit + (0 if include_self else 1)))
    values, indices = torch.topk(angles, k=k, largest=False, sorted=True)

    rows: list[dict[str, Any]] = []
    rank = 1
    for value, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        token_id = int(idx)
        if not include_self and token_id == anchor_id:
            continue
        rows.append(_neighborhood_row(assets, token_id, float(value), rank))
        rank += 1
        if len(rows) >= limit:
            break
    return rows


@torch.inference_mode()
def iter_neighborhood_csv(assets: ModelAssets, anchor_id: int) -> Iterator[str]:
    angles = _angles_for_anchor(assets, anchor_id)
    order = torch.argsort(angles, stable=True).detach().cpu().tolist()
    angles_cpu = angles.detach().cpu().tolist()
    magnitudes_cpu = assets.magnitudes.detach().cpu().tolist()

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["rank", "token_id", "angle_deg", "magnitude", "token_raw", "token_display"])
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    for rank, token_id in enumerate(order, start=1):
        info = assets.token(int(token_id))
        writer.writerow(
            [
                rank,
                int(token_id),
                float(angles_cpu[token_id]),
                float(magnitudes_cpu[token_id]),
                info.raw,
                info.display,
            ]
        )
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
