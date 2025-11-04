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
        links = super().links()
        if self.lenny_id:
            # Determine whether this record represents encrypted (loaned)
            # content. Default to False when the attribute is missing.
            encrypted = getattr(self, "is_encrypted", False)
            # Accept an optional base_url on the record (set by the
            # provider) and prefix it before the /v1/api path. Avoid
            # duplicate slashes.
            base_url = getattr(self, "base_url", "") or ""
            base_url = base_url.rstrip("/")
            path = f"/v1/api/items/{self.lenny_id}"
            base_uri = f"{base_url}{path}" if base_url else path

            # Primary acquisition link depends on encryption/loan status.
            if encrypted:
                primary = Link(
                    href=f"{base_uri}/borrow",
                    rel="http://opds-spec.org/acquisition/borrow",
                    type="application/json",
                )
                # Add a return endpoint for borrowed items
                return_link = Link(
                    href=f"{base_uri}/return",
                    rel="http://librarysimplified.org/terms/return",
                    type="application/json",
                )
                links = [primary, return_link]
            else:
                # Open/readable content served at /read
                primary = Link(
                    href=f"{base_uri}/read",
                    rel="http://opds-spec.org/acquisition/open-access",
                    type="application/json",
                )
                links = [primary]

        return links

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
    """Normalize different shapes returned by OpenLibraryDataProvider.search."""
    if isinstance(resp, tuple):
        records = resp[0] if len(resp) >= 1 else []
        total = resp[1] if len(resp) > 1 else None
        return records, total

    for attr in ("records", "docs", "items", "data"):
        if hasattr(resp, attr):
            records = getattr(resp, attr)
            total = None
            for tot_attr in ("total", "num_found", "numFound", "count", "size"):
                if hasattr(resp, tot_attr):
                    total = getattr(resp, tot_attr)
                    break
            return records, total

    try:
        records = list(resp)
        return records, None
    except Exception:
        raise TypeError(
            "cannot unpack non-iterable search response from OpenLibraryDataProvider.search"
        )


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

        # Handle the SearchResponse returned by pyopds2
        if isinstance(resp, SearchResponse):
            ol_records = resp.records or []
            total = getattr(resp, "total", None)
        else:
            ol_records, total = _unwrap_search_response(resp)

        # Accept either an indexable sequence of lenny_ids or a mapping from
        # a record key -> lenny_id. Convert non-mapping iterables to a list
        # so indexing works for dict_keys and similar types.
        lenny_ids_map = lenny_ids if isinstance(lenny_ids, Mapping) else None
        lenny_ids_list = None if lenny_ids_map else (list(lenny_ids) if lenny_ids is not None else None)

        lenny_records = []
        for idx, record in enumerate(ol_records):
            data = record.model_dump()
            # Use the exact lenny_id provided (if any). Do not use the loop
            # index as the id — prefer any existing id in the record otherwise.
            # Determine lenny_id from mapping, list, or existing data.
            assigned_id = None
            if lenny_ids_map:
                # Try to match by OpenLibrary record key (e.g. '/works/OL...')
                rec_key = data.get("key") or data.get("id")
                assigned_id = lenny_ids_map.get(rec_key)
            elif lenny_ids_list:
                if idx < len(lenny_ids_list):
                    assigned_id = lenny_ids_list[idx]

            data["lenny_id"] = (assigned_id if assigned_id is not None else data.get("lenny_id"))
            # Propagate encryption/loan status and optional base_url onto
            # the record so LennyDataRecord.links() can decide between
            # /borrow and /read endpoints and prefix the API host.
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

        base_url = (base_url or "").rstrip("/")
        def _prefix(path: str) -> str:
            return f"{base_url}{path}" if base_url else path

        return {
            "metadata": {
                "title": "Lenny Local Catalog",
                "totalItems": total,
                "itemsPerPage": limit,
                "currentOffset": offset,
            },
            "publications": publications,
            "links": [
                {"rel": "self", "href": _prefix(f"/v1/api/opds?offset={offset}&limit={limit}")},
                {
                    "rel": "next",
                    "href": _prefix(f"/v1/api/opds?offset={offset + limit}&limit={limit}"),
                },
            ],
        }
