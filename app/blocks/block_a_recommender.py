"""
Bloco A — Sistema de Recomendação
==================================

OBJETIVO
--------
Dado um usuário, devolver os k itens que ele provavelmente gostaria
mas ainda não consumiu.

INTUIÇÃO DO BASELINE
--------------------
Usamos **filtragem colaborativa item-based** com **similaridade do cosseno**.
A intuição em duas linhas:

    "Itens que tendem a ser consumidos pelos mesmos usuários são
     parecidos. Se você consumiu o item X, recomendo itens parecidos
     com X que você ainda não viu."

Passo a passo da matemática:

1. **Matriz de interações R** (usuários × itens). R[u][i] é a nota
   que o usuário u deu ao item i (ou 0 se não houve interação).

2. **Similaridade item-item S** (itens × itens). S[i][j] é o cosseno
   entre as colunas i e j de R: cos(R[:,i], R[:,j]) ∈ [-1, 1].
   Itens que foram avaliados de forma parecida pelos mesmos usuários
   ficam próximos.

3. **Score de recomendação**: para o usuário u, score(u, i) = S · R[u].
   Em palavras: somamos as similaridades do item i com TODOS os itens
   que u consumiu, ponderadas pela nota que u deu. Itens já consumidos
   são zerados (-inf) para nunca aparecerem.

4. **Cold-start**: se u é novo (não está na matriz), retornamos os
   itens mais populares globalmente. É uma heurística simples mas
   eficaz contra o problema clássico de "novo usuário sem histórico".

POR QUE ITEM-BASED E NÃO USER-BASED?
------------------------------------
Em catálogos típicos, itens são em menor número e mais estáveis que
usuários. Calcular similaridade item×item é mais barato, e o índice
não precisa ser recalculado quando entra um usuário novo.

REFINAMENTOS POSSÍVEIS (Semana 4+)
----------------------------------
- TruncatedSVD para fatorar R em componentes latentes (reduz ruído).
- ALS implícito (biblioteca `implicit`) para sinais binários.
- Métricas: Precision@k, Recall@k, MAP@k, NDCG@k.
- Filtro de novidade (penalizar itens muito populares).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from app.config import MODELS_DIR
from app.datasources.base import get_datasource
from app.schemas import RecommendationItem, RecommendResponse

MODEL_PATH: Path = MODELS_DIR / "block_a_recommender.joblib"


# =============================================================
# Treino
# =============================================================
def train() -> dict:
    """Constrói a matriz de similaridade item-item e persiste em disco.

    Lê interações da fonte ativa (CSV/BD/API), monta a matriz
    user×item, calcula a matriz de similaridade do cosseno entre
    colunas (itens) e salva tudo em um único bundle joblib.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    df = get_datasource().fetch_interactions()
    if df.empty:
        raise ValueError("DataFrame de interações vazio.")

    # ------------------------------------------------------------------
    # 1. Pivot: lista de tuplas (u, i, r) → matriz user×item
    # ------------------------------------------------------------------
    # `pivot_table` agrega múltiplas interações do mesmo (u,i) com a média.
    # `fill_value=0.0` trata "ausência de interação" como zero. Isso é uma
    # SIMPLIFICAÇÃO: rigorosamente, "não interagiu" ≠ "deu nota zero".
    # O modelo certo para sinais implícitos (cliques, views) é ALS, mas
    # para o baseline esta aproximação é aceitável.
    matrix = df.pivot_table(
        index="user_id", columns="item_id", values="rating", fill_value=0.0
    )

    # Listas paralelas para mapear índice numérico ↔ id real.
    # Precisamos delas porque a matriz NumPy não preserva os IDs.
    user_index = matrix.index.tolist()
    item_index = matrix.columns.tolist()

    # ------------------------------------------------------------------
    # 2. Similaridade item-item
    # ------------------------------------------------------------------
    # `matrix.T` é a transposta (itens × usuários). `cosine_similarity`
    # devolve uma matriz quadrada (n_itens × n_itens) onde [i][j] é a
    # similaridade entre os itens i e j.
    item_sim = cosine_similarity(matrix.T.values)

    # ------------------------------------------------------------------
    # 3. Persistência
    # ------------------------------------------------------------------
    # Tudo num só arquivo: matriz, similaridade, índices e metadados.
    # `astype(np.float32)` reduz tamanho do arquivo pela metade sem
    # perda relevante de precisão para recomendação.
    bundle = {
        "matrix": matrix.values.astype(np.float32),
        "item_sim": item_sim.astype(np.float32),
        "user_index": user_index,
        "item_index": item_index,
        "model_type": "cosine_item_based_baseline",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "n_users": len(user_index),
        "n_items": len(item_index),
        "n_interactions": int((matrix.values > 0).sum()),
        "model_type": bundle["model_type"],
    }
    print(f"[Bloco A] Modelo salvo em {MODEL_PATH}")
    print(f"[Bloco A] Métricas: {metrics}")
    return metrics


# =============================================================
# Inferência
# =============================================================
_cache: dict | None = None


def _load_bundle() -> dict:
    """Carrega o modelo do disco apenas na primeira chamada (singleton).

    Em produção isso evita ler 10 MB de joblib a cada request. Em
    desenvolvimento (uvicorn --reload), o cache é descartado a cada
    reload, então não há risco de servir um modelo "velho".
    """
    global _cache
    if _cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        _cache = joblib.load(MODEL_PATH)
    return _cache


def recommend(user_id: str, k: int = 5) -> RecommendResponse:
    """Recomenda k itens para o usuário.

    Algoritmo:
        scores = item_sim @ user_vec        # produto matriz×vetor
        scores[itens já consumidos] = -inf  # nunca recomenda repetido
        top = argsort(-scores)[:k]          # k melhores
    """
    bundle = _load_bundle()
    user_index: list = bundle["user_index"]
    item_index: list = bundle["item_index"]
    matrix: np.ndarray = bundle["matrix"]
    item_sim: np.ndarray = bundle["item_sim"]

    # ----------------------------------------------------------------
    # Cold-start: usuário desconhecido → recomenda os mais populares
    # ----------------------------------------------------------------
    if user_id not in user_index:
        popularity = matrix.sum(axis=0)          # soma das ratings por item
        top = np.argsort(-popularity)[:k]        # `-` para ordem decrescente
        items = [
            RecommendationItem(
                item_id=item_index[i],
                score=float(popularity[i]),
                motivo="popularidade (usuário novo)",
            )
            for i in top
        ]
        return RecommendResponse(user_id=user_id, recommendations=items)

    # ----------------------------------------------------------------
    # Caminho normal: scoring colaborativo
    # ----------------------------------------------------------------
    u_idx = user_index.index(user_id)
    user_vec = matrix[u_idx]

    # Aqui mora a "mágica" do baseline:
    #     scores[i] = soma_j ( item_sim[i][j] * user_vec[j] )
    # ou seja, para cada item i, somo as similaridades com tudo que o
    # usuário consumiu, ponderadas pela nota que ele deu.
    scores = item_sim @ user_vec

    # Bloqueia itens já consumidos pelo usuário.
    scores[user_vec > 0] = -np.inf

    top = np.argsort(-scores)[:k]
    items = [
        RecommendationItem(
            item_id=item_index[i],
            score=float(scores[i]),
            motivo="similaridade item-item",
        )
        for i in top
        if scores[i] != -np.inf  # garantia extra
    ]
    return RecommendResponse(user_id=user_id, recommendations=items)
