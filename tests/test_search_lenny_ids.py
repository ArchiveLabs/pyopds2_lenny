"""``LennyDataProvider.search`` must key ``lenny_id`` off each record's own
Open Library edition id.

``lenny_id`` builds every actionable link in the feed — ``/items/{id}/borrow``,
``/items/{id}/read``, ``/opds/{id}``, ``/items/{id}/return`` — and keys the
``encryption_map`` / ``borrowable_map`` lookups. Open Library does not echo back
the order of an ``edition_key:(A OR B OR ...)`` disjunction, and it drops
records that fail its cover / acquisition / access filters, so anything
positional silently hands a publication another book's links.
"""

from typing import Dict, List, Optional

import pytest

from pyopds2.provider import DataProvider

from pyopds2_lenny import (
    LennyDataProvider,
    LennyDataRecord,
    OpenLibraryDataProvider,
    OpenLibraryDataRecord,
)


# Five real Open Library editions. Open Library returns them for the
# disjunction below in *reverse* of the order they are requested in — verified
# live against openlibrary.org — which is what makes the ordering matter.
EDITIONS: Dict[int, str] = {
    37044497: "The Enchanted Castle",
    37044487: "Just So Stories",
    51733522: "Ozma of Oz",
    37044778: "The House of Mirth",
    37044726: "Pygmalion",
}
REQUESTED_IDS: List[int] = list(EDITIONS)


def _ol_record(edition_id: int) -> OpenLibraryDataRecord:
    """Build a work record surfacing ``OL{edition_id}M``, as OL's search does."""
    title = EDITIONS[edition_id]
    return OpenLibraryDataRecord.model_validate({
        "key": f"/works/OL{edition_id}W",
        "title": title,
        "editions": {
            "numFound": 1,
            "start": 0,
            "numFoundExact": True,
            "docs": [{"key": f"/books/OL{edition_id}M", "title": title}],
        },
    })


def _fake_ol_search(monkeypatch: pytest.MonkeyPatch, returned_ids: List[int]) -> None:
    """Stub ``OpenLibraryDataProvider.search`` to return exactly these editions."""
    def fake_search(query: str = "", limit: int = 50, offset: int = 0, **_):
        records = [_ol_record(edition_id) for edition_id in returned_ids]
        return DataProvider.SearchResponse(
            provider=OpenLibraryDataProvider,
            records=list(records),
            total=len(records),
            query=query,
            limit=limit,
            offset=offset,
            sort=None,
        )

    monkeypatch.setattr(OpenLibraryDataProvider, "search", staticmethod(fake_search))


def _query_for(ids: List[int]) -> str:
    return f"edition_key:({' OR '.join(f'OL{i}M' for i in ids)})"


def _search(ids: List[int], **kwargs) -> DataProvider.SearchResponse:
    return LennyDataProvider.search(query=_query_for(ids), limit=len(ids), **kwargs)


# --- the regression this module exists for --------------------------------


def test_lenny_id_follows_the_record_not_the_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shuffle OL's response order; every publication keeps its own identity.

    This is the test that fails on the positional implementation: with the
    response reversed, publication 1 was handed publication 5's ``lenny_id``,
    and therefore publication 5's borrow link, encryption flag and
    availability.
    """
    returned_ids = list(reversed(REQUESTED_IDS))
    _fake_ol_search(monkeypatch, returned_ids)

    encryption_map = {i: bool(n % 2) for n, i in enumerate(REQUESTED_IDS)}
    borrowable_map = {i: not bool(n % 3) for n, i in enumerate(REQUESTED_IDS)}

    resp = _search(
        REQUESTED_IDS,
        lenny_ids={i: i for i in REQUESTED_IDS},
        encryption_map=encryption_map,
        borrowable_map=borrowable_map,
    )

    # Each record keeps the id of the edition it actually describes.
    assert [r.lenny_id for r in resp.records] == returned_ids

    for record in resp.records:
        assert isinstance(record, LennyDataRecord)
        edition_id = record.lenny_id
        assert record.title == EDITIONS[edition_id], "title/id pairing broke"
        assert record.is_encrypted is encryption_map[edition_id]
        assert record.is_borrowable is borrowable_map[edition_id]

        # The links are the payload: they must address this book, not another.
        for link in record.links():
            if "/items/" in link.href or "/opds/" in link.href:
                assert f"/{edition_id}" in link.href, link.href


def test_dropped_record_does_not_shift_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OL omitting a record must not slide every later publication's id.

    ``OpenLibraryDataProvider.search`` filters out records with no cover or no
    acquisition options, so a short response is routine, not exotic.
    """
    returned_ids = [i for i in REQUESTED_IDS if i != 51733522]
    _fake_ol_search(monkeypatch, returned_ids)

    resp = _search(REQUESTED_IDS, lenny_ids={i: i for i in REQUESTED_IDS})

    assert [r.lenny_id for r in resp.records] == returned_ids
    assert all(r.title == EDITIONS[r.lenny_id] for r in resp.records)


def test_unrequested_record_gets_no_lenny_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record Lenny does not hold must not be given a borrow link."""
    held = REQUESTED_IDS[:2]
    _fake_ol_search(monkeypatch, REQUESTED_IDS)

    resp = _search(REQUESTED_IDS, lenny_ids={i: i for i in held})

    got = {r.lenny_id for r in resp.records if r.lenny_id is not None}
    assert got == set(held)
    for record in resp.records:
        if record.lenny_id is None:
            assert not any(
                "/items/" in link.href for link in record.links()
            ), "publication Lenny does not hold got a Lenny acquisition link"


# --- accepted shapes of the ``lenny_ids`` argument ------------------------


def test_search_assigns_provided_lenny_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary case: a ``{edition_id: edition_id}`` map, order preserved."""
    _fake_ol_search(monkeypatch, REQUESTED_IDS)

    resp = _search(REQUESTED_IDS, lenny_ids={i: i for i in REQUESTED_IDS})

    assert [r.lenny_id for r in resp.records] == REQUESTED_IDS


def test_search_handles_mapping_with_index_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``{edition_id: position}`` — a legacy caller shape — still resolves."""
    _fake_ol_search(monkeypatch, REQUESTED_IDS)

    mapping = {identifier: position
               for position, identifier in enumerate(REQUESTED_IDS, start=1)}
    resp = _search(REQUESTED_IDS, lenny_ids=mapping)

    assert [r.lenny_id for r in resp.records] == REQUESTED_IDS


def test_search_handles_mapping_with_index_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``{position: edition_id}`` — the other legacy shape — still resolves."""
    _fake_ol_search(monkeypatch, REQUESTED_IDS)

    mapping = dict(enumerate(REQUESTED_IDS))
    resp = _search(REQUESTED_IDS, lenny_ids=mapping)

    assert [r.lenny_id for r in resp.records] == REQUESTED_IDS


def test_search_handles_plain_iterable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_post_borrow_publication`` passes a bare list of edition ids."""
    _fake_ol_search(monkeypatch, REQUESTED_IDS)

    resp = _search(REQUESTED_IDS, lenny_ids=list(REQUESTED_IDS))

    assert [r.lenny_id for r in resp.records] == REQUESTED_IDS


def test_search_single_record_iterable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact ``build_post_borrow_publication`` call shape."""
    book_id = 37044778
    _fake_ol_search(monkeypatch, [book_id])

    resp = LennyDataProvider.search(
        query=f"edition_key:OL{book_id}M", limit=1, lenny_ids=[book_id]
    )

    assert [r.lenny_id for r in resp.records] == [book_id]


def test_search_without_lenny_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``lenny_ids`` at all: no Lenny identity, no Lenny links."""
    _fake_ol_search(monkeypatch, REQUESTED_IDS)

    resp = _search(REQUESTED_IDS)

    assert all(r.lenny_id is None for r in resp.records)
    assert all(r.is_encrypted is False for r in resp.records)


def test_record_without_editions_falls_back_to_its_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record that *is* an edition carries its id on ``key``."""
    book_id = 37044726

    def fake_search(query: str = "", limit: int = 50, offset: int = 0, **_):
        record = OpenLibraryDataRecord.model_validate({
            "key": f"/books/OL{book_id}M",
            "title": EDITIONS[book_id],
        })
        return DataProvider.SearchResponse(
            provider=OpenLibraryDataProvider, records=[record], total=1,
            query=query, limit=limit, offset=offset, sort=None,
        )

    monkeypatch.setattr(OpenLibraryDataProvider, "search", staticmethod(fake_search))

    resp = LennyDataProvider.search(
        query=f"edition_key:OL{book_id}M", limit=1, lenny_ids=[book_id]
    )

    assert [r.lenny_id for r in resp.records] == [book_id]


def test_work_key_is_not_mistaken_for_an_edition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/works/OL123W`` must never be read as edition 123."""
    def fake_search(query: str = "", limit: int = 50, offset: int = 0, **_):
        record = OpenLibraryDataRecord.model_validate({
            "key": "/works/OL99541W", "title": "The Enchanted Castle",
        })
        return DataProvider.SearchResponse(
            provider=OpenLibraryDataProvider, records=[record], total=1,
            query=query, limit=limit, offset=offset, sort=None,
        )

    monkeypatch.setattr(OpenLibraryDataProvider, "search", staticmethod(fake_search))

    resp = LennyDataProvider.search(query="q", limit=1, lenny_ids=[99541])

    assert resp.records[0].lenny_id is None
