from typing import List, Tuple, Optional
from collections.abc import Mapping
from pyopds2_openlibrary import OpenLibraryDataProvider, OpenLibraryDataRecord, Link
from opds2.provider import SearchResponse


class LennyDataRecord(OpenLibraryDataRecord):
    """Extends OpenLibraryDataRecord with local borrow/return links for Lenny."""

    lenny_id: Optional[int] = None
    base_url: Optional[str] = None
    is_encrypted: bool = False

    @property
    def type(self) -> str:
        return "http://schema.org/Book"

    def links(self) -> List[Link]:
        """Override acquisition links to use Lenny's API endpoints."""
        base_links = super().links() or []
        if not self.lenny_id:
            return base_links

        # Correctly prefix base_url
        base_url = (self.base_url or "").rstrip("/")
        base_uri = f"{base_url}/v1/api/items/{self.lenny_id}"

        if self.is_encrypted:
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

        # Default: open-access link
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
        lenny_ids: Optional[Mapping] = None,
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

        for record in ol_records:
            data = record.model_dump()
            olid_key = data.get("key") or data.get("edition_key") or ""
            olid_int = None
            # Extract the numeric part (e.g. /books/OL37044497M → 37044497)
            if isinstance(olid_key, str) and "OL" in olid_key:
                try:
                    olid_int = int(olid_key.split("OL")[1].split("M")[0])
                except Exception:
                    pass

            assigned_id = None
            if lenny_ids and olid_int in lenny_ids:
                assigned_id = lenny_ids[olid_int]

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
