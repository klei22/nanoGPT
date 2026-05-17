from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient


# The lightweight test environment may not have transformers installed. The app
# only needs these names at import time; model loading is monkeypatched below.
if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    fake_transformers.PreTrainedTokenizerBase = object
    sys.modules["transformers"] = fake_transformers

from app import main, model_service  # noqa: E402
from app.model_service import LocalModelInfo, TokenInfo  # noqa: E402


@dataclass
class FakeAssets:
    model_name: str = "fake-model"
    requested_device: str = "cpu"
    effective_device: str = "cpu"

    def __post_init__(self) -> None:
        self.token_infos = [
            TokenInfo(token_id=0, raw="<bos>", display="<bos>", normalized="<bos> bos"),
            TokenInfo(token_id=1, raw="▁Hello", display=" Hello", normalized="▁hello hello"),
            TokenInfo(token_id=2, raw="world", display="world", normalized="world world"),
            TokenInfo(token_id=3, raw="<0xF9>", display="<0xF9>", normalized="<0xf9> 0xf9 f9"),
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.token_infos)

    @property
    def hidden_dim(self) -> int:
        return 4

    def token(self, token_id: int) -> TokenInfo:
        if token_id < 0 or token_id >= len(self.token_infos):
            raise IndexError(f"token_id must be in [0, {len(self.token_infos) - 1}], got {token_id}")
        return self.token_infos[token_id]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_assets = FakeAssets()
    monkeypatch.setattr(main, "get_current_assets", lambda: fake_assets)
    monkeypatch.setattr(main, "load_active_model", lambda *args, **kwargs: fake_assets)
    return TestClient(main.app)


def test_homepage_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "LM-Head Angle Explorer" in response.text


def test_status_route_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called():
        raise AssertionError("status should not load model assets")

    monkeypatch.setattr(main, "get_current_assets", fail_if_called)
    monkeypatch.setattr(
        main,
        "get_model_status",
        lambda: {
            "loaded": False,
            "model_name": "configured/default",
            "requested_device": "cpu",
            "effective_device": "cpu",
            "vocab_size": 0,
            "hidden_dim": 0,
        },
    )
    response = TestClient(main.app).get("/api/status")
    assert response.status_code == 200
    assert response.json() == {
        "loaded": False,
        "model_name": "configured/default",
        "requested_device": "cpu",
        "effective_device": "cpu",
        "vocab_size": 0,
        "hidden_dim": 0,
    }


def test_search_before_model_load_returns_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "get_current_assets",
        lambda: (_ for _ in ()).throw(ValueError("No model is loaded. Enter a Hugging Face model ID and click Load model first.")),
    )
    response = TestClient(main.app).get("/api/tokens/search", params={"q": "hello"})
    assert response.status_code == 400
    assert "click Load model" in response.json()["detail"]


def test_search_route_is_not_captured_by_token_id_route(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "hello"
    assert [row["token_id"] for row in data["results"]] == [1]


def test_search_route_supports_blank_query_without_result_limit(client: TestClient) -> None:
    # Stale clients may still send limit, but the endpoint should ignore it and
    # return the complete vocabulary.
    response = client.get("/api/tokens/search", params={"q": "", "limit": 2})
    assert response.status_code == 200
    assert [row["token_id"] for row in response.json()["results"]] == [0, 1, 2, 3]


def test_search_route_has_no_limit_validation(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "", "limit": -1})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 4


def test_search_is_literal_and_not_pattern_matching(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "^F9$"})
    assert response.status_code == 200
    data = response.json()
    assert "mode" not in data
    assert data["results"] == []


def test_literal_search_still_matches_byte_alias_contents(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "F9"})
    assert response.status_code == 200
    assert [row["token_id"] for row in response.json()["results"]] == [3]


def test_explicit_token_lookup_route(client: TestClient) -> None:
    response = client.get("/api/tokens/id/2")
    assert response.status_code == 200
    assert response.json()["raw"] == "world"


def test_legacy_token_lookup_route_still_works(client: TestClient) -> None:
    response = client.get("/api/tokens/2")
    assert response.status_code == 200
    assert response.json()["raw"] == "world"


def test_model_load_endpoint_accepts_huggingface_designation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(model_name: str, requested_device: str, **kwargs):
        calls.append((model_name, requested_device, kwargs))
        return FakeAssets(model_name=model_name, requested_device=requested_device, effective_device=requested_device)

    monkeypatch.setattr(main, "load_active_model", fake_load)
    client = TestClient(main.app)

    response = client.post("/api/model/load", json={"model_name": "Qwen/Qwen3.5-0.8B-Base", "device": "cpu"})

    assert response.status_code == 200
    data = response.json()
    assert data["loaded"] is True
    assert data["model_name"] == "Qwen/Qwen3.5-0.8B-Base"
    assert data["requested_device"] == "cpu"
    assert calls == [("Qwen/Qwen3.5-0.8B-Base", "cpu", {"force_reload": False, "allow_download": True})]


def test_model_load_endpoint_rejects_blank_model_name() -> None:
    client = TestClient(main.app)
    response = client.post("/api/model/load", json={"model_name": "   "})
    assert response.status_code == 400
    assert "blank" in response.json()["detail"].lower() or "empty" in response.json()["detail"].lower()



def test_model_load_endpoint_can_disable_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(model_name: str, requested_device: str, **kwargs):
        calls.append((model_name, requested_device, kwargs))
        return FakeAssets(model_name=model_name, requested_device=requested_device, effective_device=requested_device)

    monkeypatch.setattr(main, "load_active_model", fake_load)
    client = TestClient(main.app)

    response = client.post(
        "/api/model/load",
        json={"model_name": "Qwen/Qwen3.5-0.8B-Base", "device": "cpu", "allow_download": False},
    )

    assert response.status_code == 200
    assert calls == [("Qwen/Qwen3.5-0.8B-Base", "cpu", {"force_reload": False, "allow_download": False})]


def test_available_models_route_returns_local_cache_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "list_local_models",
        lambda: [
            LocalModelInfo(
                model_name="Qwen/Qwen3.5-0.8B-Base",
                cache_path="/tmp/hf/models--Qwen--Qwen3.5-0.8B-Base",
                size_bytes=1234,
                last_modified=100.0,
            )
        ],
    )
    client = TestClient(main.app)

    response = client.get("/api/models/available")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "model_name": "Qwen/Qwen3.5-0.8B-Base",
                "cache_path": "/tmp/hf/models--Qwen--Qwen3.5-0.8B-Base",
                "size_bytes": 1234,
                "last_modified": 100.0,
            }
        ]
    }



def test_manual_local_cache_scanner_finds_huggingface_model_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "hub"
    (cache_dir / "models--Qwen--Qwen3.5-0.8B-Base").mkdir(parents=True)
    (cache_dir / "models--google--gemma-3-270m").mkdir(parents=True)
    (cache_dir / "datasets--not--a-model").mkdir(parents=True)

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setattr(model_service, "_scan_cache_with_hub", lambda records: None)

    models = model_service.list_local_models()

    assert [item.model_name for item in models] == [
        "google/gemma-3-270m",
        "Qwen/Qwen3.5-0.8B-Base",
    ]
