from typing import List, Tuple, Optional
from collections.abc import Mapping, Iterable
from pyopds2_openlibrary import OpenLibraryDataProvider, OpenLibraryDataRecord, Link
from pyopds2.provider import DataProvider


class LennyDataRecord(OpenLibraryDataRecord):
    """Extends OpenLibraryDataRecord with local borrow/return links for Lenny."""

    lenny_id: Optional[int] = None

    @property
    def type(self) -> str:
        return "http://schema.org/Book"

    def links(self) -> List[Link]:
        """Override acquisition links to use Lenny's API endpoints.

        If the record was created with an `is_encrypted` flag the primary
        acquisition link will be `/borrow` (for encrypted/loaned content),
        otherwise `/read` for open-access/readable content. When encrypted
        we also include a `return` endpoint.
        """
        base_links = super().links() or []
        if not self.lenny_id:
            return base_links

        # Ensure base_url is correctly prefixed
        base_url = (getattr(self, "base_url", "") or "").rstrip("/")
        base_uri = f"{base_url}/v1/api/items/{self.lenny_id}" if base_url else f"/v1/api/items/{self.lenny_id}"

        if getattr(self, "is_encrypted", False):
            return [
                Link(
                    href=f"{base_uri}/borrow",
                    rel="http://opds-spec.org/acquisition/borrow",
                    type="application/json",
                ),
                Link(
                    href=f"{base_uri}/return",
                    rel="http://librarysimplified.org/terms/return",
                    type="application/json",
                ),
            ]

        return [
            Link(
                href=f"{base_uri}/read",
                rel="http://opds-spec.org/acquisition/open-access",
                type="application/json",
            )
        ]

    def images(self) -> Optional[List[Link]]:
        """Provide cover image link based on Open Library cover ID."""
        if hasattr(self, "cover_i") and self.cover_i:
            return [
                Link(
                    href=f"https://covers.openlibrary.org/b/id/{self.cover_i}-L.jpg",
                    rel="http://opds-spec.org/image",
                    type="image/jpeg",
                )
            ]
        return None


def _unwrap_search_response(resp):
    """Minimal normalizer for the upstream search return shapes."""
    if isinstance(resp, tuple):
        records = resp[0] if len(resp) >= 1 else []
        total = resp[1] if len(resp) > 1 else None
        return records, total

    if hasattr(resp, "records"):
        return getattr(resp, "records"), getattr(resp, "total", None)

    try:
        return list(resp), None
    except TypeError:
        raise TypeError("cannot unpack non-iterable search response")


class LennyDataProvider(OpenLibraryDataProvider):
    """Adapts Open Library metadata for Lenny's local catalog."""

    @staticmethod
    def search(
        query: str,
        numfound: int,
        limit: int,
        offset: int,
        lenny_ids: Optional[Mapping[int, int]] = None,
        is_encrypted: Optional[bool] = False,
        base_url: Optional[str] = None,
    ) -> Tuple[List[LennyDataRecord], int]:
        """Perform a metadata search and adapt results into LennyDataRecords."""
        resp = OpenLibraryDataProvider.search(query=query, limit=limit, offset=offset)

        if isinstance(resp, DataProvider):
            ol_records = resp.records or []
            total = getattr(resp, "total", None)
        else:
            ol_records, total = _unwrap_search_response(resp)

        lenny_records: List[LennyDataRecord] = []

        # Convert keys to a predictable list order for mapping
        if isinstance(lenny_ids, Mapping):
            keys = list(lenny_ids.keys())
            values = list(lenny_ids.values())

            def _looks_like_index_sequence(seq: List[int]) -> bool:
                if not seq or not all(isinstance(item, int) for item in seq):
                    return False
                return seq == list(range(len(seq))) or seq == list(range(1, len(seq) + 1))

            keys_are_indices = _looks_like_index_sequence(keys)
            values_are_indices = _looks_like_index_sequence(values)

            if values and not values_are_indices:
                lenny_id_values = values
            elif keys and not keys_are_indices:
                lenny_id_values = keys
            elif values and not keys:
                lenny_id_values = values
            elif keys:
                lenny_id_values = keys
            else:
                lenny_id_values = []
        elif isinstance(lenny_ids, Iterable) and not isinstance(lenny_ids, (str, bytes)):
            lenny_id_values = list(lenny_ids)
        else:
            lenny_id_values = []

        for idx, record in enumerate(ol_records):
            data = record.model_dump()

            # Assign lenny_id properly from mapping keys
            if idx < len(lenny_id_values):
                data["lenny_id"] = lenny_id_values[idx]

            data["is_encrypted"] = bool(is_encrypted)
            data["base_url"] = base_url
            lenny_records.append(LennyDataRecord.model_validate(data))

        return lenny_records, (total if total is not None else numfound)
        
    @staticmethod
    def search_response(
        query: str,
        numfound: int,
        limit: int,
        offset: int,
        lenny_ids: Optional[Mapping[int, int]] = None,
        is_encrypted: Optional[bool] = False,
        base_url: Optional[str] = None,
    ):
        """Return a `DataProvider.SearchResponse` for consumers that expect it.

        This preserves the original `search` behavior (which returns a
        (records, total) tuple) while offering an adapter that returns the
        richer SearchResponse dataclass used by `pyopds2.Catalog` and other
        consumers.
        """
        records, total = LennyDataProvider.search(
            query=query,
            numfound=numfound,
            limit=limit,
            offset=offset,
            lenny_ids=lenny_ids,
            is_encrypted=is_encrypted,
            base_url=base_url,
        )

        return DataProvider.SearchResponse(
            provider=LennyDataProvider,
            query=query,
            limit=limit,
            offset=offset,
            sort=None,
            records=records,
            total=total,
        )

    @staticmethod
    def create_opds_feed(
        records: List[LennyDataRecord],
        total: int,
        limit: int,
        offset: int,
        base_url: Optional[str] = None,
        title: str = "Lenny Catalog",
    ):
        """Construct an OPDS 2.0 JSON feed for Lenny's books.

        The function attempts to produce JSON-serializable structures: any
        Publication models are converted to dicts via `model_dump` when
        available (pydantic v2). Navigation links include standard OPDS
        rels (self, start, previous, next, first, last).
        """
        # Convert Publication models to JSON-friendly dicts
        publications = []
        for record in records:
            pub = record.to_publication()
            try:
                publications.append(pub.model_dump())
            except Exception:
                publications.append(pub)

        base = (base_url or "").rstrip("/")

        def _href(path: str) -> str:
            return f"{base}{path}" if base else path

        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))
        safe_total = max(0, int(total))

        last_offset = 0
        if safe_total and safe_limit:
            last_page_index = (safe_total - 1) // safe_limit
            last_offset = last_page_index * safe_limit

        links: List[dict] = []
        links.append({"rel": "self", "href": _href(f"/v1/api/opds?offset={safe_offset}&limit={safe_limit}")})
        links.append({"rel": "start", "href": _href("/v1/api/opds")})

        if safe_offset > 0:
            prev_offset = max(0, safe_offset - safe_limit)
            links.append({"rel": "previous", "href": _href(f"/v1/api/opds?offset={prev_offset}&limit={safe_limit}")})

        if safe_offset + safe_limit < safe_total:
            next_offset = safe_offset + safe_limit
            links.append({"rel": "next", "href": _href(f"/v1/api/opds?offset={next_offset}&limit={safe_limit}")})

        if safe_total:
            if safe_offset != 0:
                links.append({"rel": "first", "href": _href(f"/v1/api/opds?offset=0&limit={safe_limit}")})
            if last_offset and safe_offset != last_offset:
                links.append({"rel": "last", "href": _href(f"/v1/api/opds?offset={last_offset}&limit={safe_limit}")})

        return {
            "metadata": {
                "title": title,
                "totalItems": safe_total,
                "itemsPerPage": safe_limit,
                "currentOffset": safe_offset,
            },
            "publications": publications,
            "links": links,
        }
