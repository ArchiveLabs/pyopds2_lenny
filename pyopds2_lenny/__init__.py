# pyopds_lenny/__init__.py
from typing import List, Tuple, Optional
from collections.abc import Mapping
from pyopds2_openlibrary import OpenLibraryDataProvider, OpenLibraryDataRecord, Link
from opds2.provider import SearchResponse


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
        # Start with any links provided by the OpenLibrary record.
        base_links = super().links() or []
        if not self.lenny_id:
            return base_links

        # Small helper to build the base item URI consistently.
        base_url = (getattr(self, "base_url", "") or "").rstrip("/")
        if base_url:
            base_uri = f"{base_url}/v1/api/items/{self.lenny_id}"
        else:
            base_uri = f"/v1/api/items/{self.lenny_id}"

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

        # Default: open-access/readable content
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
    """Minimal normalizer for the upstream search return shapes.

    Keep this small: accept (records, total) tuples, objects with
    `records` and optional `total`, or any iterable of records.
    """
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
        lenny_ids: Optional[List[int]] = None,
        is_encrypted: Optional[bool] = False,
        base_url: Optional[str] = None,
    ) -> Tuple[List[LennyDataRecord], int]:
        """Perform a metadata search and adapt results into LennyDataRecords."""
        resp = OpenLibraryDataProvider.search(query=query, limit=limit, offset=offset)

        if isinstance(resp, SearchResponse):
            ol_records = resp.records or []
            total = getattr(resp, "total", None)
        else:
            ol_records, total = _unwrap_search_response(resp)

        # Accept a mapping {record_key: lenny_id} or a sequence [id1, id2, ...].
        lenny_ids_map = lenny_ids if isinstance(lenny_ids, Mapping) else None
        lenny_ids_list = None if lenny_ids_map else (list(lenny_ids) if lenny_ids is not None else None)

        lenny_records: List[LennyDataRecord] = []
        for idx, record in enumerate(ol_records):
            data = record.model_dump()

            assigned_id = None
            if lenny_ids_map:
                rec_key = data.get("key") or data.get("id")
                if rec_key is not None:
                    assigned_id = lenny_ids_map.get(rec_key)
            elif lenny_ids_list and idx < len(lenny_ids_list):
                assigned_id = lenny_ids_list[idx]

            # If a lenny id was provided, prefer that; otherwise keep any
            # existing `lenny_id` the record might already carry.
            if assigned_id is not None:
                data["lenny_id"] = assigned_id

            data["is_encrypted"] = bool(is_encrypted)
            data["base_url"] = base_url
            lenny_records.append(LennyDataRecord.model_validate(data))

        return lenny_records, (total if total is not None else numfound)

    @staticmethod
    def create_opds_feed(
        records: List[LennyDataRecord],
        total: int,
        limit: int,
        offset: int,
        base_url: Optional[str] = None,
    ):
        """Construct an OPDS 2.0 JSON feed for Lenny's books.

        If a `base_url` is provided, prefix it to the feed-level links so
        consumers receive fully-qualified URLs instead of relative paths.
        """
        publications = [record.to_publication() for record in records]

        base = (base_url or "").rstrip("/")
        def _href(path: str) -> str:
            return f"{base}{path}" if base else path

        return {
            "metadata": {
                "title": "Lenny Local Catalog",
                "totalItems": total,
                "itemsPerPage": limit,
                "currentOffset": offset,
            },
            "publications": publications,
            "links": [
                {"rel": "self", "href": _href(f"/v1/api/opds?offset={offset}&limit={limit}")},
                {"rel": "next", "href": _href(f"/v1/api/opds?offset={offset + limit}&limit={limit}")},
            ],
        }
