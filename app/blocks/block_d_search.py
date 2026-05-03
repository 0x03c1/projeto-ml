"""
Bloco D — Busca Semântica
==========================

OBJETIVO
--------
Substituir a clássica `WHERE descricao LIKE '%termo%'` por uma busca
que entende SIGNIFICADO. Quem digita "computador para programar"
encontra "notebook leve para desenvolvedores" mesmo sem nenhuma
palavra em comum.

INTUIÇÃO DO BASELINE (TF-IDF)
-----------------------------
TF-IDF vetoriza cada documento em um espaço de palavras. Cada palavra
recebe um peso:

    tf(palavra, doc) = quantas vezes a palavra aparece no documento
    idf(palavra)     = log(N / n_docs_com_a_palavra)
    tfidf            = tf * idf

A intuição: palavras raras na coleção (alto IDF) carregam mais
informação que palavras comuns ("o", "de", "para"). A similaridade
entre uma query e um documento é o cosseno entre seus vetores TF-IDF.

POR QUE TF-IDF COMO BASELINE?
    1. Não exige GPU nem download de modelo.
    2. Roda em milissegundos para corpus de até ~100k itens.
    3. É uma BASELINE FORTE — em buscas com vocabulário compartilhado
       entre query e documento, dificilmente algum modelo neural
       supera TF-IDF + bigramas em mais que alguns pontos.

LIMITAÇÃO PRINCIPAL
-------------------
TF-IDF não captura sinônimos. "computador" e "notebook" são vetores
ortogonais para ele. É justamente onde os embeddings entram.

REFINAMENTO RECOMENDADO (Semana 4+)
-----------------------------------
Substituir TF-IDF por **sentence-transformers**. O modelo
`paraphrase-multilingual-MiniLM-L12-v2` mapeia frases em PT-BR para
vetores de 384 dimensões em um espaço onde semântica próxima fica
geometricamente próxima. Para corpus grandes, use FAISS para
indexação aproximada.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from app.config import MODELS_DIR
from app.datasources.base import get_datasource
from app.schemas import SearchHit, SearchResponse

INDEX_PATH: Path = MODELS_DIR / "block_d_search.joblib"


# =============================================================
# Construção do índice
# =============================================================
def build_index() -> dict:
    """Constrói o índice TF-IDF e persiste em INDEX_PATH.

    Note que para Bloco D não temos "treino" propriamente dito — o
    `vectorizer.fit_transform` apenas aprende o vocabulário e os IDFs.
    Não há rótulo, não há holdout: avaliamos com queries qualitativas
    e métricas como Recall@k em uma amostra rotulada manualmente.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = get_datasource().fetch_corpus()
    if df.empty:
        raise ValueError("Corpus vazio.")

    # ----------------------------------------------------------------
    # 1. Concatena título + texto em um único campo "documento"
    # ----------------------------------------------------------------
    # Em produção, é comum dar mais peso ao título (multiplicar por k
    # antes de concatenar, ou usar um vetorizador por campo).
    # Aqui mantemos simples para o baseline.
    df["__doc__"] = (
        df["titulo"].fillna("") + " " + df["texto"].fillna("")
    ).str.lower()

    # ----------------------------------------------------------------
    # 2. Vetorização TF-IDF
    # ----------------------------------------------------------------
    # ngram_range=(1,2): considera unigramas E bigramas.
    #   Captura "máquina virtual" como token único, separado de "máquina"
    #   e "virtual". Aumenta a discriminação para corpus técnicos.
    # max_features=20000: limita o vocabulário aos 20k tokens mais
    #   frequentes. Mantém o índice leve e remove ruído.
    # min_df=1: aceita tokens que aparecem em ao menos 1 doc. Em corpus
    #   maiores (>10k itens), considere min_df=2 para descartar typos.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(df["__doc__"].tolist())

    # ----------------------------------------------------------------
    # 3. Persistência
    # ----------------------------------------------------------------
    bundle = {
        "vectorizer": vectorizer,
        "matrix": matrix,                                # esparsa (scipy.sparse)
        "items": df[["item_id", "titulo"]].to_dict("records"),
        "engine": "tfidf_baseline",
    }
    joblib.dump(bundle, INDEX_PATH)

    metrics = {
        "n_items": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "engine": bundle["engine"],
    }
    print(f"[Bloco D] Índice salvo em {INDEX_PATH}")
    print(f"[Bloco D] Métricas: {metrics}")
    return metrics


# =============================================================
# Busca
# =============================================================
_cache: dict | None = None


def _load_bundle() -> dict:
    global _cache
    if _cache is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(INDEX_PATH)
        _cache = joblib.load(INDEX_PATH)
    return _cache


def search(query: str, k: int = 10) -> SearchResponse:
    """Retorna os k itens mais próximos da query.

    Algoritmo:
        1. Vetoriza a query usando o MESMO vectorizer do índice
           (vocabulário e IDFs aprendidos no fit).
        2. Calcula cosseno entre query e cada item do corpus.
        3. Retorna os k mais altos com score > 0.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    bundle = _load_bundle()
    vectorizer = bundle["vectorizer"]
    matrix = bundle["matrix"]
    items = bundle["items"]

    # Importante: usar `.transform`, NÃO `.fit_transform`. O vocabulário
    # já foi fixado no build_index. Se você refitar agora, perde o
    # alinhamento com a matriz indexada.
    q_vec = vectorizer.transform([query.lower()])

    # cosine_similarity entre 1×V e N×V devolve 1×N. Pegamos a linha [0].
    scores = cosine_similarity(q_vec, matrix)[0]

    # `argsort(-x)` ordena decrescente. `[:k]` pega os k primeiros.
    top = np.argsort(-scores)[:k]

    hits = [
        SearchHit(
            item_id=items[i]["item_id"],
            titulo=items[i]["titulo"],
            score=float(scores[i]),
        )
        for i in top
        if scores[i] > 0  # filtra itens sem nenhuma sobreposição lexical
    ]
    return SearchResponse(query=query, hits=hits)
