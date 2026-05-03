"""
Contrato comum a todas as fontes de dados.
============================================

Todas as fontes (CSV, banco de dados, API REST) implementam o mesmo
protocolo. Isso permite trocar a origem dos dados em UMA única linha
de configuração (`DATASOURCE_KIND` em `app/config.py`) sem precisar
mexer nos blocos de ML.

Por que `Protocol` em vez de `ABC`?
-----------------------------------
Em Python moderno (3.8+), `typing.Protocol` permite "duck typing
estático": qualquer classe que tenha os métodos certos satisfaz o
protocolo, mesmo sem herdar dele explicitamente. Isso deixa o código
mais simples — não precisamos de uma hierarquia de classes.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    """Toda fonte de dados expõe estes métodos.

    Cada bloco chama apenas o método que lhe interessa:
        Bloco A  -> fetch_interactions()
        Bloco B  -> fetch_dataset()
        Bloco C  -> fetch_texts()    (opcional; muitos blocos C usam só o request)
        Bloco D  -> fetch_corpus()
    """

    def fetch_interactions(self) -> pd.DataFrame:
        """Retorna DataFrame com colunas [user_id, item_id, rating]."""
        ...

    def fetch_dataset(self) -> pd.DataFrame:
        """Retorna DataFrame com features + target. Colunas variam por domínio."""
        ...

    def fetch_texts(self) -> pd.DataFrame:
        """Retorna DataFrame com colunas [id, texto, label?]."""
        ...

    def fetch_corpus(self) -> pd.DataFrame:
        """Retorna DataFrame com colunas [item_id, titulo, texto]."""
        ...


def get_datasource() -> DataSource:
    """Factory: devolve a fonte de dados ativa segundo `app.config.DATASOURCE_KIND`.

    Esta é a única função que os blocos precisam conhecer. Toda a
    decisão de "de onde vêm os dados" acontece aqui.
    """
    from app.config import DATASOURCE_KIND

    if DATASOURCE_KIND == "csv":
        from app.datasources.local_csv import LocalCSVSource
        return LocalCSVSource()

    if DATASOURCE_KIND == "database":
        from app.datasources.database import DatabaseSource
        return DatabaseSource()

    if DATASOURCE_KIND == "api":
        from app.datasources.public_api import PublicApiSource
        return PublicApiSource()

    raise ValueError(
        f"DATASOURCE_KIND desconhecido: {DATASOURCE_KIND!r}. "
        "Use 'csv', 'database' ou 'api'."
    )
