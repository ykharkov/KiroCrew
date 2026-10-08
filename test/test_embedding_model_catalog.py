"""First-class embedding models: the shipped catalog, per-model profiles and their download."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from kiro_crew import asset_downloader as dl
from kiro_crew import embedding_catalog as catalog
from kiro_crew import embeddings as emb

_HEX = set("0123456789abcdef")


def test_every_shipped_entry_loads_with_a_pinned_identity() -> None:
    shipped = json.loads(catalog._CATALOG_FILE.read_text(encoding="utf-8"))["models"]
    entries = catalog.embedding_model_catalog()
    assert [entry.id for entry in entries] == [item["id"] for item in shipped]
    for entry in entries:
        assert len(entry.sha256) == 64 and set(entry.sha256) <= _HEX
        assert len(entry.revision) == 40 and set(entry.revision) <= _HEX
        assert entry.url == (
            f"https://huggingface.co/{entry.repo}/resolve/{entry.revision}/{entry.file}"
        )
        assert entry.size_bytes > emb._GGUF_MIN_BYTES and entry.dim > 0


@pytest.mark.parametrize("catalog_id", ["custom", "custom@mean", "a:b", ""])
def test_a_catalog_id_cannot_be_a_label_the_loader_writes(catalog_id: str) -> None:
    # A catalog id becomes an identity label, so it may not collide with the
    # custom labels whose pooling the loader reads from the file itself.
    item = json.loads(catalog._CATALOG_FILE.read_text(encoding="utf-8"))["models"][0]
    with pytest.raises(ValueError):
        catalog._catalog_model({**item, "id": catalog_id})


def test_a_catalog_profile_applies_only_to_its_pinned_identity() -> None:
    entry = catalog.embedding_model_catalog()[0]
    assert emb.embedding_profile(entry.model_id) == entry.profile
    # Its label over other bytes, and its bytes under the plain custom label,
    # keep the default profile: neither names the space the entry's vectors are in.
    assert emb.embedding_profile(f"{entry.id}:sha256:{'0' * 64}") == emb.EmbeddingProfile()
    assert emb.embedding_profile(f"custom:sha256:{entry.sha256}") == emb.EmbeddingProfile()
    assert emb.embedding_profile("qwen3-embedding:0.6b") == emb.EmbeddingProfile()


def test_queries_and_documents_take_the_model_prompts() -> None:
    entry = next(entry for entry in catalog.embedding_model_catalog() if entry.profile.query_prefix)
    prompted = emb.LlamaCppEmbedder(
        model_path=Path("/nonexistent.gguf"), dim=entry.dim, model_id=entry.model_id
    )
    plain = emb.LlamaCppEmbedder(model_path=Path("/nonexistent.gguf"))
    assert emb.prompted_text(prompted, "x", query=True) == entry.profile.query_prefix + "x"
    assert emb.prompted_text(prompted, "x") == entry.profile.document_prefix + "x"
    assert emb.prompted_text(plain, "x", query=True) == "x"
    assert emb.prompted_text(object(), "x", query=True) == "x"


def test_the_loader_runs_the_model_with_its_pooling_and_bos(tmp_path: Path, monkeypatch) -> None:
    entry = next(entry for entry in catalog.embedding_model_catalog() if not entry.profile.add_bos)
    bos: list[bool] = []

    class _FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
            bos.append(add_bos)
            return [1]

        def n_embd(self) -> int:
            return entry.dim

    monkeypatch.setattr(emb, "_load_llama_class", lambda: _FakeLlama)
    model = tmp_path / entry.file
    model.write_bytes(b"\0" * (emb._GGUF_MIN_BYTES + 1))
    embedder = emb.LlamaCppEmbedder(model_path=model, dim=entry.dim, model_id=entry.model_id)
    try:
        assert embedder.wait_ready(timeout=5)
        llm = embedder._llm
        assert llm.kwargs["pooling_type"] == entry.profile.pooling_type  # type: ignore[attr-defined]
        llm.tokenize(b"text")  # type: ignore[attr-defined]
    finally:
        embedder.close()
    assert bos == [False]


@pytest.mark.parametrize(
    ("target", "followed"),
    [
        ("https://us.aws.cdn.hf.co/xet-bridge-us/abc", True),
        ("https://huggingface.co/other/path", True),
        ("https://evil-hf.co/x", False),
        ("https://cdn.hf.co.example.com/x", False),
        ("http://cdn.hf.co/x", False),
        ("https://cdn.hf.co:8443/x", False),
    ],
)
def test_a_catalog_transfer_follows_only_hugging_face_redirects(
    target: str, followed: bool
) -> None:
    handler = dl._DomainAllowlistRedirectHandler(catalog._HF_REDIRECT_DOMAINS)
    request = urllib.request.Request("https://huggingface.co/r/m/resolve/rev/f.gguf")
    try:
        verdict = handler.redirect_request(request, None, 302, "Found", {}, target)
    except dl.RedirectRefused:
        verdict = None
    assert (verdict is not None) == followed
