from pathlib import Path

import pytest

from research_intake.arxiv import MalformedPage, parse_atom, safe_user_agent


def test_parse_valid_atom_and_malformed_entry(mixed_xml: bytes) -> None:
    page = parse_atom(mixed_xml)
    assert len(page.entries) == 3
    assert len(page.malformed_entries) == 1
    assert page.total_results == 4


def test_author_order_and_cross_lists(mixed_xml: bytes) -> None:
    entry = parse_atom(mixed_xml).entries[0]
    assert entry["authors"] == ["Alice Smith", "Bob Jones"]
    assert entry["categories"] == ["q-fin.TR", "q-fin.ST"]


def test_source_doi_and_links(mixed_xml: bytes) -> None:
    entry = parse_atom(mixed_xml).entries[0]
    assert entry["doi"] == "10.1000/example"
    assert any(link.get("title") == "pdf" for link in entry["links"])


def test_malformed_document_is_page_failure(project_root: Path) -> None:
    body = (project_root / "tests" / "fixtures" / "arxiv" / "malformed.xml").read_bytes()
    with pytest.raises(MalformedPage):
        parse_atom(body)


def test_invalid_total_results_is_page_failure() -> None:
    body = b"""<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>unknown</opensearch:totalResults></feed>"""
    with pytest.raises(MalformedPage, match="totalResults"):
        parse_atom(body)


def test_optional_contact_user_agent() -> None:
    assert safe_user_agent("app/1", None) == "app/1"
    assert "ops@example.test" in safe_user_agent("app/1", "ops@example.test")
