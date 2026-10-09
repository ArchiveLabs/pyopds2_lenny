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
