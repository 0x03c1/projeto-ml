"""
Fonte de dados: Banco de dados relacional (Grupo 1).
=====================================================

Este adapter conecta o microsserviço de ML diretamente ao banco do
projeto integrador. Útil quando os dados mudam com frequência e exportar
CSV manualmente toda semana se torna inviável.

Como ativar:
    1. Instale o driver:
        pip install sqlalchemy psycopg2-binary    # PostgreSQL
        pip install sqlalchemy pymysql            # MySQL/MariaDB
        pip install sqlalchemy                    # SQLite

    2. Defina a variável de ambiente:
        export ML_DATABASE_URL="postgresql://user:pass@host:5432/banco"

    3. Em `app/config.py` mude:
        DATASOURCE_KIND = "database"

    4. Adapte as queries SQL deste arquivo ao schema real do projeto.

Por que SQLAlchemy?
    Abstrai o driver concreto, então o mesmo código serve PostgreSQL,
    MySQL e SQLite. Isso permite que a equipe use SQLite localmente
    (rápido, sem servidor) e PostgreSQL em produção.

Segurança:
    NUNCA hardcode senha no código. Sempre via variável de ambiente.
    Adicione `.env` ao `.gitignore`.
"""
from __future__ import annotations

import os

import pandas as pd


class DatabaseSource:
    """Conecta no BD do projeto integrador via SQLAlchemy.

    A engine é lazy: só conecta de fato quando uma query é executada.
    """

    def __init__(self) -> None:
        url = os.environ.get("ML_DATABASE_URL")
        if not url:
            raise RuntimeError(
                "Variável de ambiente ML_DATABASE_URL não definida. "
                "Exemplo: postgresql://user:pass@localhost:5432/projeto_integrador"
            )
        # Import tardio: SQLAlchemy só é exigido quando este adapter é
        # realmente instanciado. Equipes que usam CSV/API não precisam
        # instalar nada extra.
        from sqlalchemy import create_engine
        self.engine = create_engine(url, pool_pre_ping=True)

    # =============================================================
    # Bloco A — interações usuário×item
    # =============================================================
    def fetch_interactions(self) -> pd.DataFrame:
        # TODO equipe: ajuste a query ao schema real.
        # O DataFrame de saída deve ter EXATAMENTE estas 3 colunas.
        sql = """
            SELECT
                usuario_id  AS user_id,
                produto_id  AS item_id,
                nota        AS rating
            FROM avaliacoes
            WHERE nota IS NOT NULL
        """
        return pd.read_sql(sql, self.engine)

    # =============================================================
    # Bloco B — features + target
    # =============================================================
    def fetch_dataset(self) -> pd.DataFrame:
        # TODO equipe: substituir pelo SELECT real do domínio.
        sql = """
            SELECT
                idade,
                frequencia,
                media,
                evadiu AS target
            FROM alunos
        """
        return pd.read_sql(sql, self.engine)

    # =============================================================
    # Bloco C — textos rotulados
    # =============================================================
    def fetch_texts(self) -> pd.DataFrame:
        sql = """
            SELECT
                id,
                comentario AS texto,
                rotulo     AS label
            FROM avaliacoes_textuais
        """
        return pd.read_sql(sql, self.engine)

    # =============================================================
    # Bloco D — corpus de busca
    # =============================================================
    def fetch_corpus(self) -> pd.DataFrame:
        sql = """
            SELECT
                id          AS item_id,
                nome        AS titulo,
                descricao   AS texto
            FROM produtos
            WHERE ativo = TRUE
        """
        return pd.read_sql(sql, self.engine)
