from collections import OrderedDict
from types import SimpleNamespace

import pytest

from pyopds2_lenny import LennyDataProvider, LennyDataRecord, OpenLibraryDataProvider


class _DummyRecord:
    def __init__(self, idx: int) -> None:
        # Provide minimal payload expected by model_dump
        self._payload = {
            "title": f"Test Title {idx}",
            "key": f"OL{idx}M",
        }

    def model_dump(self):
        return dict(self._payload)


def _setup_search(monkeypatch: pytest.MonkeyPatch, requested_ids):
    dummy_records = [_DummyRecord(idx) for idx in range(len(requested_ids))]
    captured_payloads = []

    def fake_search(*_, **__):
        return dummy_records, len(dummy_records)

    def fake_model_validate(cls, data):
        captured_payloads.append(data)
        return SimpleNamespace(**data)

    monkeypatch.setattr(OpenLibraryDataProvider, "search", staticmethod(fake_search))
    monkeypatch.setattr(LennyDataRecord, "model_validate", classmethod(fake_model_validate))

    return captured_payloads
def test_search_assigns_provided_lenny_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_ids = [37044497, 37044487, 51733522, 37044778, 37044726]
    captured_payloads = _setup_search(monkeypatch, requested_ids)

    mapping = OrderedDict((index, identifier) for index, identifier in enumerate(requested_ids))

    records, total = LennyDataProvider.search(
        query="test",
        numfound=len(requested_ids),
        limit=len(requested_ids),
        offset=0,
        lenny_ids=mapping,
        is_encrypted=False,
        base_url="https://example.org",
    )

    assert total == len(requested_ids)
    assert [record.lenny_id for record in records] == requested_ids
    assert [payload["lenny_id"] for payload in captured_payloads] == requested_ids

    # Ensure base metadata remains intact for downstream consumers
    assert all(payload["base_url"] == "https://example.org" for payload in captured_payloads)
    assert all(payload["is_encrypted"] is False for payload in captured_payloads)


def test_search_handles_mapping_with_index_values(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_ids = [37044497, 37044487, 51733522, 37044778, 37044726]
    captured_payloads = _setup_search(monkeypatch, requested_ids)

    # Simulate lenny_ids mapping where values are mere index counters
    mapping = OrderedDict((identifier, position) for position, identifier in enumerate(requested_ids, start=1))

    records, _ = LennyDataProvider.search(
        query="test",
        numfound=len(requested_ids),
        limit=len(requested_ids),
        offset=0,
        lenny_ids=mapping,
        is_encrypted=False,
        base_url=None,
    )

    assert [record.lenny_id for record in records] == requested_ids
    assert [payload["lenny_id"] for payload in captured_payloads] == requested_ids


def test_search_handles_dict_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_ids = [37044497, 37044487, 51733522, 37044778, 37044726]
    captured_payloads = _setup_search(monkeypatch, requested_ids)

    mapping = OrderedDict((identifier, None) for identifier in requested_ids)

    records, _ = LennyDataProvider.search(
        query="test",
        numfound=len(requested_ids),
        limit=len(requested_ids),
        offset=0,
        lenny_ids=mapping.keys(),
        is_encrypted=False,
        base_url=None,
    )

    expected = list(mapping.keys())
    assert [record.lenny_id for record in records] == expected
    assert [payload["lenny_id"] for payload in captured_payloads] == expected
