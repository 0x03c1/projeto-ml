"""
Modelos Pydantic de request/response.
======================================

Este módulo define o **contrato** entre a aplicação cliente (web do Grupo 1
ou notebook/dashboard do Grupo 2) e o microsserviço de ML.

Por que usar Pydantic?
----------------------
1. **Validação automática** — se o cliente enviar um payload mal-formado,
   o FastAPI já rejeita com HTTP 422 antes de o código do bloco rodar.
2. **Documentação OpenAPI gratuita** — os exemplos em `json_schema_extra`
   aparecem direto em `/docs` (Swagger UI), sem código adicional.
3. **Tipagem forte** — o IDE autocompleta os campos e o linter pega
   acessos errados como `response.recomendation` (sem 's' no final).

Convenção: cada bloco usa um conjunto próprio de schemas. Se o seu
projeto exigir campos adicionais (ex.: `idade` e `renda` para o Bloco B),
crie schemas específicos aqui em vez de "engordar" `dict[str, Any]`.
"""
from typing import Any
from pydantic import BaseModel, Field


# =============================================================
# Saúde do serviço
# =============================================================
class HealthResponse(BaseModel):
    """Resposta do endpoint /health.

    A aplicação web pode chamar este endpoint a cada N segundos para
    saber se o microsserviço está vivo. Um cliente bem-comportado
    cacheia esta resposta por alguns segundos para não sobrecarregar
    o serviço.
    """
    status: str
    service: str
    version: str
    bloco_ativo: str | None = None
    datasource: str | None = None  # "csv" | "database" | "api"


# =============================================================
# Bloco A — Recomendação
# =============================================================
class RecommendationItem(BaseModel):
    """Um item recomendado.

    O campo `motivo` é opcional, mas ALTAMENTE recomendado: ele dá
    explicabilidade ("você está vendo este item porque curtiu X").
    Sistemas de recomendação sem explicação geram desconfiança no
    usuário final.
    """
    item_id: str
    score: float
    motivo: str | None = None


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]


# =============================================================
# Bloco B — Classificação / Predição
# =============================================================
class PredictRequest(BaseModel):
    """Entrada do endpoint /predict.

    O dicionário `features` é deliberadamente flexível porque cada
    equipe modela um problema diferente: para evasão estudantil seria
    `{"idade": 22, "frequencia": 0.65, "media": 6.3}`; para detecção
    de fraude seria `{"valor": 1234.56, "horario": 3, "categoria": "viagem"}`.

    A validação dos nomes esperados (FEATURE_COLUMNS) é feita dentro do
    bloco, não aqui. Isso permite que o mesmo schema sirva os 4 cenários
    do Grupo 2 (IBGE, PRF, INEP, etc.).
    """
    features: dict[str, Any] = Field(
        ...,
        description="Features de entrada do modelo, no formato chave/valor.",
        json_schema_extra={
            "example": {"idade": 22, "frequencia": 0.65, "media": 6.3}
        },
    )


class PredictResponse(BaseModel):
    """Saída do endpoint /predict.

    `prediction` é o rótulo previsto (string ou número).
    `proba` é o vetor de probabilidades por classe — útil quando a
    aplicação web precisa exibir confiança ("85% chance de evadir").
    Modelos de regressão (saída contínua) deixam `proba=None`.
    """
    prediction: Any
    proba: dict[str, float] | None = None


# =============================================================
# Bloco C — Texto / Sentimento
# =============================================================
class TextRequest(BaseModel):
    """Entrada do endpoint /analyze.

    O limite de 5000 caracteres protege o serviço de DoS por payload
    gigante. Se o seu domínio precisar analisar textos maiores
    (artigos completos, por exemplo), aumente o limite e documente
    a justificativa no relatório.
    """
    text: str = Field(..., min_length=1, max_length=5000)


class TextAnalysisResponse(BaseModel):
    """Saída do endpoint /analyze.

    `sentimento` segue a convenção {"positivo", "negativo", "neutro"}
    para que a aplicação web possa pintar um badge de cor sem precisar
    de mapeamento adicional.

    `extras` carrega informações de debug (engine usada, contagens,
    probabilidades brutas). Em produção, pode ser omitido.
    """
    sentimento: str
    score: float
    extras: dict[str, Any] | None = None


# =============================================================
# Bloco D — Busca Semântica
# =============================================================
class SearchHit(BaseModel):
    """Um resultado da busca.

    `score` é a similaridade do cosseno entre a query e o item, no
    intervalo [0, 1] para TF-IDF e tipicamente [-1, 1] para
    sentence-transformers (mas na prática quase sempre positivo).
    """
    item_id: str
    titulo: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


# =============================================================
# Erro padrão
# =============================================================
class ErrorResponse(BaseModel):
    """Schema usado quando o FastAPI gera HTTPException.

    Manter um schema de erro consistente facilita o tratamento no
    frontend (sempre dá pra ler `response.detail`).
    """
    detail: str
