from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

ARXIV_VERSION = re.compile(r"(?i)(v\d+)$")


def split_arxiv_identifier(value: str) -> tuple[str, str]:
    candidate = value.strip()
    if "://" in candidate:
        candidate = urlparse(candidate).path.rsplit("/abs/", 1)[-1].lstrip("/")
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate[6:]
    match = ARXIV_VERSION.search(candidate)
    if not match:
        raise ValueError(f"arXiv identifier lacks terminal version: {value!r}")
    version = match.group(1).lower()
    base = candidate[: match.start()].strip().lower()
    if not base:
        raise ValueError("empty base arXiv identifier")
    return base, version


def item_id_for(base_arxiv_id: str) -> str:
    canonical_key = f"arxiv:{base_arxiv_id.lower()}"
    return "ri_" + hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()


def version_number(version: str) -> int:
    return int(version[1:])
