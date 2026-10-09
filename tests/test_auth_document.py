from pyopds2_lenny import LennyDataProvider


def test_auth_doc_without_issuer_is_implicit_only(monkeypatch):
    monkeypatch.setattr(LennyDataProvider, "BASE_URL", "http://x/")
    monkeypatch.setattr(LennyDataProvider, "OAUTH_ISSUER", "")
    auth = LennyDataProvider.get_authentication_document()["authentication"]
    assert [a["type"] for a in auth] == ["http://opds-spec.org/auth/oauth/implicit"]


def test_auth_doc_with_issuer_adds_pkce(monkeypatch):
    monkeypatch.setattr(LennyDataProvider, "BASE_URL", "http://x/")
    monkeypatch.setattr(LennyDataProvider, "OAUTH_ISSUER", "https://l.example/")
    auth = LennyDataProvider.get_authentication_document()["authentication"]
    assert auth[0]["type"] == "http://opds-spec.org/auth/oauth/implicit"
    assert auth[1]["type"] == "http://opds-spec.org/auth/oauth/authorization-code-with-pkce"
    assert {l["rel"]: l["href"] for l in auth[1]["links"]} == {
        "authenticate": "https://l.example/v1/api/oauth2/authorize",
        "code": "https://l.example/v1/api/oauth2/token",
        "refresh": "https://l.example/v1/api/oauth2/token",
    }


def _doc(monkeypatch, issuer, flows=None):
    monkeypatch.setattr(LennyDataProvider, "BASE_URL", "http://x/")
    monkeypatch.setattr(LennyDataProvider, "OAUTH_ISSUER", issuer)
    d = LennyDataProvider.get_authentication_document(flows)
    return d["id"], [a["type"].rsplit("/", 1)[-1] for a in d["authentication"]]


def test_flows_none_is_legacy(monkeypatch):
    assert _doc(monkeypatch, "") == ("http://x/oauth/implicit", ["implicit"])
    assert _doc(monkeypatch, "https://l") == ("http://x/oauth/implicit", ["implicit", "authorization-code-with-pkce"])


def test_flows_selection_and_id(monkeypatch):
    pkce = "authorization-code-with-pkce"
    assert _doc(monkeypatch, "https://l", ["implicit"]) == ("http://x/oauth/implicit", ["implicit"])
    assert _doc(monkeypatch, "https://l", ["pkce"]) == ("http://x/oauth/authentication", [pkce])
    assert _doc(monkeypatch, "https://l", ["pkce", "implicit"]) == ("http://x/oauth/implicit", ["implicit", pkce])


def test_flows_unavailable_falls_back_to_implicit(monkeypatch):
    assert _doc(monkeypatch, "", ["pkce"]) == ("http://x/oauth/implicit", ["implicit"])
    assert _doc(monkeypatch, "https://l", []) == ("http://x/oauth/implicit", ["implicit"])
