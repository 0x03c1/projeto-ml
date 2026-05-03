"""
Fonte de dados: arquivos CSV locais.
=====================================

É a fonte mais simples e a recomendada para EDA (Semana 2). A equipe
exporta uma vez do banco de dados (ou da API) e trabalha em cima do
arquivo, sem depender de rede ou credenciais.

Convenção de nomes (em `data/`):
    interactions.csv  → Bloco A (user_id, item_id, rating)
    dataset.csv       → Bloco B (features + target)
    texts.csv         → Bloco C (id, texto, label?)
    corpus.csv        → Bloco D (item_id, titulo, texto)

Quando o arquivo não existe, retornamos um pequeno dataset sintético
para o pipeline rodar de cabo a rabo logo no primeiro `python train.py`.
Isso evita o anti-padrão "instalei tudo, rodei e deu erro de arquivo
não encontrado" que costuma travar a Semana 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import DATA_DIR, SEED


class LocalCSVSource:
    """Lê CSVs em `data/`. Cai em sintético se o arquivo não existir."""

    # ---------- Bloco A ----------
    def fetch_interactions(self) -> pd.DataFrame:
        path = DATA_DIR / "interactions.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df[["user_id", "item_id", "rating"]]
        return self._synthetic_interactions()

    # ---------- Bloco B ----------
    def fetch_dataset(self) -> pd.DataFrame:
        path = DATA_DIR / "dataset.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._synthetic_dataset()

    # ---------- Bloco C ----------
    def fetch_texts(self) -> pd.DataFrame:
        path = DATA_DIR / "texts.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._synthetic_texts()

    # ---------- Bloco D ----------
    def fetch_corpus(self) -> pd.DataFrame:
        path = DATA_DIR / "corpus.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df[["item_id", "titulo", "texto"]]
        return self._synthetic_corpus()

    # =============================================================
    # Datasets sintéticos (fallback)
    # =============================================================
    @staticmethod
    def _synthetic_interactions() -> pd.DataFrame:
        print("[CSV] AVISO: usando interações sintéticas (data/interactions.csv ausente).")
        rng = np.random.default_rng(SEED)
        rows = []
        for u in range(50):
            n_int = int(rng.integers(3, 12))
            items = rng.choice(30, size=n_int, replace=False)
            for i in items:
                rows.append({
                    "user_id": f"u{u:03d}",
                    "item_id": f"i{i:03d}",
                    "rating": float(rng.integers(1, 6)),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _synthetic_dataset() -> pd.DataFrame:
        print("[CSV] AVISO: usando dataset sintético (data/dataset.csv ausente).")
        rng = np.random.default_rng(SEED)
        n = 500
        df = pd.DataFrame({
            "feature_1": rng.normal(0, 1, n),
            "feature_2": rng.normal(0, 1, n),
            "feature_3": rng.normal(0, 1, n),
        })
        df["target"] = (df["feature_1"] + df["feature_2"] - df["feature_3"] > 0).astype(int)
        return df

    @staticmethod
    def _synthetic_texts() -> pd.DataFrame:
        print("[CSV] AVISO: usando textos sintéticos (data/texts.csv ausente).")
        return pd.DataFrame([
            {"id": "t1", "texto": "Adorei o atendimento, muito rápido!", "label": "positivo"},
            {"id": "t2", "texto": "Produto chegou quebrado, péssimo.", "label": "negativo"},
            {"id": "t3", "texto": "Entrega no prazo, sem comentários.", "label": "neutro"},
        ])

    @staticmethod
    def _synthetic_corpus() -> pd.DataFrame:
        print("[CSV] AVISO: usando corpus sintético (data/corpus.csv ausente).")
        temas = [
            ("Notebook", "computador portátil leve para trabalho e estudo"),
            ("Smartphone", "celular com câmera de alta resolução e bateria duradoura"),
            ("Cafeteira elétrica", "aparelho doméstico para preparar café expresso"),
            ("Livro de receitas", "coleção de receitas culinárias da gastronomia brasileira"),
            ("Mochila escolar", "mochila resistente com compartimentos para livros"),
            ("Fone de ouvido", "fone com cancelamento de ruído ativo e boa qualidade sonora"),
            ("Câmera fotográfica", "câmera digital com lente intercambiável"),
            ("Tênis de corrida", "tênis leve para corridas de longa distância"),
            ("Smart TV", "televisão com aplicativos de streaming integrados"),
            ("Liquidificador", "eletrodoméstico para preparar sucos e vitaminas"),
        ]
        return pd.DataFrame([
            {"item_id": f"i{i:03d}", "titulo": t, "texto": x}
            for i, (t, x) in enumerate(temas)
        ])
