from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"


class MalformedPage(ValueError):
    pass


@dataclass(frozen=True)
class ParsedPage:
    entries: list[dict[str, Any]]
    malformed_entries: list[dict[str, Any]]
    total_results: int


def _text(element: ET.Element, path: str, default: str | None = None) -> str | None:
    found = element.find(path)
    return found.text if found is not None and found.text is not None else default


def parse_atom(body: bytes) -> ParsedPage:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise MalformedPage(str(error)) from error
    total_text = _text(root, f"{{{OPENSEARCH}}}totalResults", "0") or "0"
    try:
        total_results = int(total_text)
    except ValueError as error:
        raise MalformedPage("invalid opensearch:totalResults") from error
    xml_language = "{http://www.w3.org/XML/1998/namespace}lang"
    feed_language = root.attrib.get(xml_language, "und")
    entries: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for index, entry in enumerate(root.findall(f"{{{ATOM}}}entry")):
        try:
            record_id = (_text(entry, f"{{{ATOM}}}id") or "").strip()
            title = (_text(entry, f"{{{ATOM}}}title") or "").strip()
            abstract = (_text(entry, f"{{{ATOM}}}summary") or "").strip()
            published = (_text(entry, f"{{{ATOM}}}published") or "").strip()
            updated = (_text(entry, f"{{{ATOM}}}updated") or "").strip()
            primary = entry.find(f"{{{ARXIV}}}primary_category")
            primary_category = primary.attrib.get("term", "") if primary is not None else ""
            authors = [(_text(author, f"{{{ATOM}}}name") or "").strip() for author in entry.findall(f"{{{ATOM}}}author")]
            categories = [category.attrib.get("term", "") for category in entry.findall(f"{{{ATOM}}}category")]
            if not all((record_id, title, abstract, published, updated, primary_category)) or not authors:
                raise ValueError("missing required Atom entry field")
            try:
                datetime.fromisoformat(published.replace("Z", "+00:00"))
                datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("invalid Atom entry timestamp") from error
            links = [dict(link.attrib) for link in entry.findall(f"{{{ATOM}}}link")]
            entries.append(
                {
                    "source_record_id": record_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": categories,
                    "primary_category": primary_category,
                    "published": published,
                    "updated": updated,
                    "doi": (_text(entry, f"{{{ARXIV}}}doi") or "").strip() or None,
                    "journal_ref": (_text(entry, f"{{{ARXIV}}}journal_ref") or "").strip() or None,
                    "comments": (_text(entry, f"{{{ARXIV}}}comment") or "").strip() or None,
                    "license": (_text(entry, f"{{{ARXIV}}}license") or "").strip() or None,
                    "language": entry.attrib.get(xml_language, feed_language),
                    "links": links,
                }
            )
        except ValueError as error:
            malformed.append({"entry_index": index, "message": str(error)})
    return ParsedPage(entries, malformed, total_results)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def request_page(
    client: httpx.Client,
    endpoint: str,
    params: dict[str, str | int],
    headers: dict[str, str],
    timeout: float,
) -> httpx.Response:
    return client.get(endpoint, params=params, headers=headers, timeout=timeout)


def selected_headers(headers: httpx.Headers) -> dict[str, str]:
    selected = {}
    for name in ("Content-Type", "Content-Encoding", "ETag", "Last-Modified", "Retry-After"):
        if name in headers:
            selected[name] = headers[name]
    return selected


def safe_user_agent(base: str, email: str | None) -> str:
    return f"{base} (contact: {email})" if email else base
