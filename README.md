# 📘 pyopds2-lenny — OPDS 2.0 Feed Adapter for Lenny

`pyopds2-lenny` is a Python library that extends [`pyopds2_openlibrary`](https://github.com/ArchiveLabs/pyopds2_openlibrary) to produce **Lenny-compatible OPDS 2.0 feeds**. It provides metadata enrichment, borrowing links, and feed-generation utilities that make it easy to serve digital library catalogs in the **OPDS 2.0** format.

> Designed for developers integrating **Open Library metadata** into **Lenny** or any custom OPDS 2.0-compliant distribution system.

---

## ✨ Key Features

* 📚 **OPDS 2.0-Compliant Feeds** — Easily generate and serve valid OPDS JSON feeds.
* 🔗 **Local Borrowing Actions** — Adds `/borrow`, `/return`, or `/read` endpoints for Lenny-hosted content.
* 🧠 **Metadata Normalization** — Merges Open Library data into structured OPDS publication records.
* 🖼️ **Cover Image Support** — Automatically attaches cover links from Open Library.
* 🧩 **Plug-and-Play Integration** — Drop it into existing `pyopds2` or FastAPI-based backends.

---

## 📦 Installation

```bash
pip install pyopds2-lenny
```

Or from source:

```bash
pip install git+https://github.com/ArchiveLabs/pyopds2_lenny.git
```

**Requirements**:

* Python ≥ 3.12
* Dependencies: `pyopds2`, `pyopds2_openlibrary`, `pydantic>=2.0`

---

## ⚙️ Example: Building an OPDS Feed

```python
from pyopds2_lenny import LennyDataProvider

# Fetch Open Library data and wrap with Lenny's borrowable records
records, total = LennyDataProvider.search(
    query="data science",
    numfound=0,
    limit=10,
    offset=0,
    lenny_ids=[101, 102, 103],
    is_encrypted=True,
    base_url="https://catalog.mylibrary.org",
)

# Generate a valid OPDS 2.0 feed
feed = LennyDataProvider.create_opds_feed(
    records=records,
    total=total,
    limit=10,
    offset=0,
    base_url="https://catalog.mylibrary.org",
)

# Serialize to JSON and serve via your web framework
import json
print(json.dumps(feed, indent=2))
```

---

## 🆔 Mapping Lenny IDs

Each OPDS entry can be tied to your local Lenny `item` table. The provider ensures correct mapping and ordering.

```python
ids = {0: 40001, 1: 40002, 2: 40003}

records, _ = LennyDataProvider.search(
    query="library",
    numfound=len(ids),
    limit=len(ids),
    offset=0,
    lenny_ids=ids,
    is_encrypted=False,
)

for record in records:
    print(record.lenny_id)
```

✅ Works with lists, dicts, and iterables.

---

## 🔐 Borrowing vs Reading Links

| `is_encrypted` | Generated Links      | Use Case                             |
| -------------- | -------------------- | ------------------------------------ |
| `True`         | `/borrow`, `/return` | DRM or restricted access titles      |
| `False`        | `/read`              | Public domain or open-access content |

All links are resolved against your configured `base_url` and conform to [OPDS 2.0 Acquisition spec](https://specs.opds.io/acquisition.html).

---

## 🧪 Testing

```bash
pip install -e .[dev]
pytest -v
```

Includes tests for:

* OPDS feed serialization
* Record field normalization
* Lenny ID mapping logic

---

## 🧑‍💻 Development Workflow

```bash
git clone https://github.com/ArchiveLabs/pyopds2_lenny.git
cd pyopds2_lenny
python -m venv venv && source venv/bin/activate
pip install -e .[dev]
pytest --cov
```

Optionally, run code style checks:

```bash
ruff check .
black .
```

---

## 📄 License

Licensed under the **AGPL-3.0 license**.
See [LICENSE](./LICENSE) for more information.

---

## 🌍 Related Libraries

* [**pyopds2**](https://github.com/ArchiveLabs/pyopds2) — Core OPDS 2.0 feed framework.
* [**pyopds2_openlibrary**](https://github.com/ArchiveLabs/pyopds2_openlibrary) — Open Library adapter.
* [**Lenny**](https://lennyforlibraries.org) — Digital library platform for modern archives.
