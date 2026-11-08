from research_intake.identity import item_id_for, split_arxiv_identifier, version_number


def test_version_stripping() -> None:
    assert split_arxiv_identifier("2607.01234v12") == ("2607.01234", "v12")


def test_url_and_legacy_identifier() -> None:
    assert split_arxiv_identifier("https://arxiv.org/abs/hep-th/9901001v2") == ("hep-th/9901001", "v2")


def test_identity_stable_across_versions() -> None:
    first, _ = split_arxiv_identifier("2607.01234v1")
    second, _ = split_arxiv_identifier("2607.01234v2")
    assert item_id_for(first) == item_id_for(second)


def test_different_papers_have_different_ids() -> None:
    assert item_id_for("2607.1") != item_id_for("2607.2")


def test_numeric_version() -> None:
    assert version_number("v12") == 12
