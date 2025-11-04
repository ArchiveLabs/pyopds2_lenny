# pyopds_lenny/__init__.py
from typing import List, Tuple, Optional
from pyopds2_openlibrary import OpenLibraryDataProvider, OpenLibraryDataRecord, Link 


class LennyDataRecord(OpenLibraryDataRecord):
    """A record that extends OpenLibrary metadata with local Lenny borrow links."""

    lenny_id: Optional[int] = None
    
    @property
    def type(self) -> str:
        return "http://schema.org/Book"
    
    def links(self) -> List[Link]:
        """Override acquisition links to point to Lenny's endpoints."""
        links = super().links()
        if self.lenny_id:
            # Replace Open Library borrow/download links with Lenny local links
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
    
    def images(self) -> List[Link]:
        if hasattr(self, "cover_i") and self.cover_i:
            return [Link(
            href=f"https://covers.openlibrary.org/b/id/{self.cover_i}-L.jpg",
            rel="http://opds-spec.org/image",
            type="image/jpeg",
            )]
        return None


class LennyDataProvider(OpenLibraryDataProvider):
    """A DataProvider that adapts Open Library metadata for Lenny's local catalog."""

    @staticmethod
    def search(
        query: str,
        numfound: int,
        limit: int,
        offset: int,
        lenny_ids: Optional[List[int]] = None,
    ) -> Tuple[List[LennyDataRecord], int]:
        """
        Perform a metadata search using Open Library's data provider,
        given a pre-computed query from Lenny.
        """
        # Use OpenLibraryDataProvider to fetch enriched metadata
        ol_records, _ = OpenLibraryDataProvider.search(
            query=query,
            limit=limit,
            offset=offset,
        )

        # Adapt OpenLibraryDataRecords to LennyDataRecords
        lenny_records = []
        for idx, record in enumerate(ol_records):
            data = record.model_dump()
            data["lenny_id"] = lenny_ids[idx] if lenny_ids and idx < len(lenny_ids) else None
            lenny_records.append(LennyDataRecord.model_validate(data))

        return lenny_records, numfound