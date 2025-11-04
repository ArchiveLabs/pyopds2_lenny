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

        if isinstance(resp, SearchResponse):
            ol_records = resp.records or []
            total = getattr(resp, "total", None)
        else:
            ol_records, total = _unwrap_search_response(resp)

        lenny_records: List[LennyDataRecord] = []

        # Convert keys to a predictable list order for mapping
        if isinstance(lenny_ids, Mapping):
            lenny_id_values = list(lenny_ids.keys())
        elif isinstance(lenny_ids, list):
            lenny_id_values = lenny_ids
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
    def create_opds_feed(
        records: List[LennyDataRecord],
        total: int,
        limit: int,
        offset: int,
        base_url: Optional[str] = None,
    ):
        """Construct an OPDS 2.0 JSON feed for Lenny's books."""
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
