#!/usr/bin/env python3
"""FastAPI/Plotly dashboard for inspecting Hugging Face normalization gains."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import apply_norm_for_display, effective_gain

BASE = Path(__file__).parent
app = FastAPI(title="LayerNorm Channel Explorer")
STATE: dict[str, object] = {}


class LoadRequest(BaseModel):
	model_name: str
	device: str = "cpu"
	allow_download: bool = True


def discover_norms(model) -> dict[str, torch.nn.Module]:
	"""Find LayerNorm/RMSNorm-like modules with a one-dimensional gain."""
	return {
		name: module for name, module in model.named_modules()
		if getattr(module, "weight", None) is not None
		and module.weight.ndim == 1
		and ("norm" in name.casefold() or "norm" in type(module).__name__.casefold())
	}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
	return (BASE / "index.html").read_text(encoding="utf-8")


@app.post("/api/load")
def load(request: LoadRequest):
	kwargs = {"local_files_only": not request.allow_download}
	try:
		model = AutoModelForCausalLM.from_pretrained(request.model_name, **kwargs).to(request.device).eval()
		tokenizer = AutoTokenizer.from_pretrained(request.model_name, **kwargs)
	except Exception as exc:
		raise HTTPException(400, f"Could not load model: {exc}") from exc
	STATE.update(model=model, tokenizer=tokenizer, norms=discover_norms(model), model_name=request.model_name)
	return {"model_name": request.model_name, "norms": list(STATE["norms"]), "vocab_size": model.get_input_embeddings().num_embeddings}


@app.get("/api/plot")
def plot(norm: str, token_id: int = 0, embedding: str = "input"):
	if "model" not in STATE:
		raise HTTPException(400, "Load a model first")
	model = STATE["model"]
	norms = STATE["norms"]
	if norm not in norms:
		raise HTTPException(404, "Unknown normalization module")
	embed_module = model.get_output_embeddings() if embedding == "output" else model.get_input_embeddings()
	weight = embed_module.weight.detach()
	if token_id < 0 or token_id >= weight.shape[0]:
		raise HTTPException(400, "token_id is outside the embedding vocabulary")
	gain = effective_gain(norms[norm]).float().cpu()
	vector = weight[token_id].detach().float().cpu()
	if vector.numel() != gain.numel():
		raise HTTPException(400, "Selected embedding and normalization have different channel counts")
	order = torch.argsort(gain, descending=True)
	after = apply_norm_for_display(norms[norm], vector)
	return {
		"norm": norm, "token_id": token_id, "embedding": embedding,
		"channel": order.tolist(), "gain": gain[order].tolist(),
		"before": vector[order].tolist(), "after": after[order].tolist(),
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
