"""
Testes de fumaça (smoke tests).
================================

São testes mínimos que verificam se o serviço sobe e responde aos
endpoints. Não testam qualidade do modelo — isso é feito via métricas
no relatório.

Rode com:
    pytest tests/
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    """O endpoint /health sempre devolve 200 e status 'ok'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["bloco_ativo"] in {"A", "B", "C", "D"}


def test_endpoints_inativos_retornam_404():
    """Endpoints de blocos não-ativos devem retornar 404."""
    from app.main import BLOCO_ATIVO

    if BLOCO_ATIVO != "A":
        r = client.get("/recommend?user_id=x&k=3")
        assert r.status_code == 404

    if BLOCO_ATIVO != "B":
        r = client.post("/predict", json={"features": {}})
        assert r.status_code == 404

    if BLOCO_ATIVO != "C":
        r = client.post("/analyze", json={"text": "teste"})
        assert r.status_code == 404

    if BLOCO_ATIVO != "D":
        r = client.get("/search?q=teste&k=3")
        assert r.status_code == 404

def test_search_query_vazia_devolve_400():
    """Requisições mal formadas no Bloco D devolvem 400."""
    from app.main import BLOCO_ATIVO

    if BLOCO_ATIVO != "D":
        return  # endpoint nem ativo

    resp = client.get("/search", params={"q": "   "})
    assert resp.status_code == 400
