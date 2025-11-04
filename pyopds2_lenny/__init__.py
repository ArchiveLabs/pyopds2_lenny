# pyopds_lenny/__init__.py
from typing import List, Tuple, Optional
from pyopds2_openlibrary import OpenLibraryDataProvider, OpenLibraryDataRecord, Link
from opds2.models import SearchResponse


class LennyDataRecord(OpenLibraryDataRecord):
    """Extends OpenLibraryDataRecord with local borrow/return links for Lenny."""

    lenny_id: Optional[int] = None

    @property
    def type(self) -> str:
        return "http://schema.org/Book"

    def links(self) -> List[Link]:
        """Override acquisition links to use Lenny's API endpoints."""
        links = super().links()
        if self.lenny_id:
            base_uri = f"/v1/api/items/{self.lenny_id}"
            links = [
                Link(
                    href=f"{base_uri}/borrow",
                    rel="http://opds-spec.org/acquisition/borrow",
                    type="application/json",
                ),
                Link(
                    href=f"{base_uri}/return",
                    rel="http://opds-spec.org/acquisition/borrow",
                    type="application/json",
                ),
            ]
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
    ) -> Tuple[List[LennyDataRecord], int]:
        """Perform a metadata search and adapt results into LennyDataRecords."""
        resp = OpenLibraryDataProvider.search(query=query, limit=limit, offset=offset)

        # Handle the SearchResponse returned by pyopds2
        if isinstance(resp, SearchResponse):
            ol_records = resp.records or []
            total = getattr(resp, "total", None)
        else:
            ol_records, total = _unwrap_search_response(resp)

        lenny_records = []
        for idx, record in enumerate(ol_records):
            data = record.model_dump()
            data["lenny_id"] = (
                lenny_ids[idx] if lenny_ids and idx < len(lenny_ids) else None
            )
            lenny_records.append(LennyDataRecord.model_validate(data))

        return lenny_records, (total if total is not None else numfound)

    @staticmethod
    def create_opds_feed(records: List[LennyDataRecord], total: int, limit: int, offset: int):
        """Construct an OPDS 2.0 JSON feed for Lenny's books."""
        publications = [record.to_publication() for record in records]

        return {
            "metadata": {
                "title": "Lenny Local Catalog",
                "totalItems": total,
                "itemsPerPage": limit,
                "currentOffset": offset,
            },
            "publications": publications,
            "links": [
                {"rel": "self", "href": f"/v1/api/opds?offset={offset}&limit={limit}"},
                {
                    "rel": "next",
                    "href": f"/v1/api/opds?offset={offset + limit}&limit={limit}",
                },
            ],
        }
