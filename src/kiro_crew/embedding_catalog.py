"""First-class embedding models: the pinned catalog and its download.

``data/embedding_models.json`` lists the models measured for Kiro Crew, each pinned
to one Hugging Face file by repo, commit, sha256 and size, with the profile it was
measured with. :func:`kiro_crew.embeddings.embedding_profile` applies an entry's
profile only to its exact id, ``<entry id>:sha256:<pinned digest>``. The Memory
tab downloads an entry into the models directory beside the bundled model and
applies it through the dashboard's ordinary model change.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
from pathlib import Path
from typing import NamedTuple

from kiro_crew import asset_downloader
from kiro_crew import embeddings as _embeddings
from kiro_crew.embeddings import EmbeddingProfile

logger = logging.getLogger(__name__)

_CATALOG_FILE = Path(__file__).resolve().parent / "data" / "embedding_models.json"


class CatalogModel(NamedTuple):
    """A first-class model from ``data/embedding_models.json``, pinned to one file."""

    id: str
    name: str
    repo: str
    revision: str
    file: str
    sha256: str
    size_bytes: int
    dim: int
    license: str
    license_url: str
    profile: EmbeddingProfile

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"

    @property
    def model_id(self) -> str:
        return f"{self.id}:sha256:{self.sha256}"

    @property
    def path(self) -> Path:
        return _embeddings.models_dir() / self.file


def _catalog_model(item: dict) -> CatalogModel:
    entry = CatalogModel(
        id=str(item["id"]),
        name=str(item["name"]),
        repo=str(item["repo"]),
        revision=str(item["revision"]),
        file=str(item["file"]),
        sha256=str(item["sha256"]),
        size_bytes=int(item["size_bytes"]),
        dim=int(item["dim"]),
        license=str(item["license"]),
        license_url=str(item["license_url"]),
        profile=EmbeddingProfile(
            _embeddings._POOLING_TYPES[item["pooling"]],
            bool(item.get("add_bos", True)),
            str(item.get("query_prefix", "")),
            str(item.get("document_prefix", "")),
        ),
    )
    # The file lands in the models directory beside the bundled model, and the id
    # becomes an identity label, so neither may escape its slot.
    if (
        not entry.id
        or ":" in entry.id
        or _embeddings._POOLING_LABEL_SEPARATOR in entry.id
        or entry.id in _embeddings._LOADER_LABELS
        or Path(entry.file).name != entry.file
        or not entry.file.endswith(".gguf")
        or entry.file == _embeddings._GGUF_FILENAME
    ):
        raise ValueError(f"invalid embedding model catalog entry {entry.id!r}")
    return entry


@functools.lru_cache(maxsize=1)
def embedding_model_catalog() -> tuple[CatalogModel, ...]:
    """The first-class models, in display order; empty when the file is unusable.

    An entry's profile applies to exactly one model id, ``<id>:sha256:<digest>``,
    which is also the id that names its vector space. Changing an entry's
    pooling, prompts or BOS handling therefore needs a new ``id``: the old one
    would go on naming vectors that were produced the old way.
    """
    try:
        raw = json.loads(_CATALOG_FILE.read_text(encoding="utf-8"))
        return tuple(_catalog_model(item) for item in raw["models"])
    except (OSError, ValueError, KeyError, TypeError):
        logger.error("The embedding model catalog is unusable", exc_info=True)
        return ()


def catalog_model(catalog_id: str) -> CatalogModel | None:
    """The catalog entry named *catalog_id*, or ``None``."""
    return next((entry for entry in embedding_model_catalog() if entry.id == catalog_id), None)


# ── First-class model download ──

# Hugging Face answers a resolve url on huggingface.co with a redirect to its
# storage CDN under hf.co; a catalog transfer may follow it there and nowhere else.
_HF_REDIRECT_DOMAINS = ("huggingface.co", "hf.co")
_catalog_download_lock = threading.Lock()
_catalog_download: dict[str, object] = {"id": "", "step": "idle", "error": ""}


def catalog_model_downloaded(entry: CatalogModel) -> bool:
    """True when *entry*'s file is installed at its pinned size.

    Its digest was verified when it was downloaded and is verified again when it
    is applied, so this cheap check is only what the Memory tab displays.
    """
    try:
        return entry.path.stat().st_size == entry.size_bytes
    except OSError:
        return False


def catalog_download_status() -> dict[str, object]:
    """Snapshot of the one catalog download: ``id``, ``step``, ``error`` and byte counts."""
    with _catalog_download_lock:
        return dict(_catalog_download)


def _update_catalog_download(**fields: object) -> None:
    global _catalog_download
    with _catalog_download_lock:
        _catalog_download = {**_catalog_download, **fields}


def start_catalog_download(entry: CatalogModel) -> bool:
    """Download *entry* into the models directory on a daemon thread.

    Returns False when a catalog download is already running. One at a time:
    each is up to a gigabyte, and the Memory tab shows a single progress bar.
    """
    global _catalog_download
    with _catalog_download_lock:
        if _catalog_download["step"] in ("downloading", "verifying"):
            return False
        _catalog_download = {"id": entry.id, "step": "downloading", "error": ""}
    threading.Thread(
        target=_download_catalog_model, args=(entry,), name="kc-catalog-download", daemon=True
    ).start()
    return True


def _download_catalog_model(entry: CatalogModel) -> None:
    if os.environ.get(_embeddings._SKIP_DOWNLOAD_ENV) == "1":
        _update_catalog_download(step="failed", error=f"{_embeddings._SKIP_DOWNLOAD_ENV}=1 is set")
        return
    try:
        entry.path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _update_catalog_download(step="failed", error=f"models directory unavailable: {exc}")
        return
    # resume=True: the file is up to a gigabyte, and only the gateway downloads
    # into this staging name, so a transfer cut off by a restart continues.
    ok, error = asset_downloader.download_to(
        entry.path,
        entry.url,
        sha256=entry.sha256,
        size=entry.size_bytes,
        min_bytes=_embeddings._GGUF_MIN_BYTES,
        resume=True,
        timeout_secs=_embeddings._HTTP_TIMEOUT_SECS,
        chunk_bytes=_embeddings._HTTP_CHUNK_BYTES,
        progress_every_bytes=_embeddings._PROGRESS_EVERY_BYTES,
        on_progress=lambda done, total: _update_catalog_download(
            bytes_downloaded=done, bytes_total=total
        ),
        on_verifying=lambda: _update_catalog_download(step="verifying"),
        redirect_domains=_HF_REDIRECT_DOMAINS,
        label=f"embedding model {entry.id}",
    )
    if ok:
        logger.info("Embedding model %s installed at %s", entry.id, entry.path)
    else:
        logger.warning("Embedding model %s download failed: %s", entry.id, error)
    _update_catalog_download(step="ready" if ok else "failed", error=error)
