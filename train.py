"""
Entry point de treino.
=======================

Uso:
    python train.py            # treina o BLOCO_ATIVO definido em app/main.py
    python train.py --bloco A  # força um bloco específico
    python train.py --bloco D --datasource api  # ad-hoc para Grupo 2

Por que centralizar tudo aqui?
------------------------------
Cada bloco tem seu próprio `train()`. Este script é só uma fina
camada de roteamento que escolhe qual chamar e gravar as métricas
em `models/metrics.json`. Manter um único entry point evita confusão
("rodei qual script mesmo?") e facilita a automação (CI/CD).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from app.config import METRICS_PATH


def _train_bloco(bloco: str) -> dict:
    bloco = bloco.upper()
    if bloco == "A":
        from app.blocks import block_a_recommender
        return block_a_recommender.train()
    if bloco == "B":
        from app.blocks import block_b_classifier
        return block_b_classifier.train()
    if bloco == "C":
        # Bloco C não tem treino próprio se usar baseline lexical ou
        # modelo pré-treinado. Se a equipe treinar um classificador
        # supervisionado por cima de embeddings, coloque a chamada aqui.
        print("[Bloco C] Sem treino: bloco usa modelo pré-treinado / léxico.")
        return {"engine": "lexicon_or_pretrained"}
    if bloco == "D":
        from app.blocks import block_d_search
        return block_d_search.build_index()
    raise ValueError(f"Bloco desconhecido: {bloco}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o bloco escolhido.")
    parser.add_argument(
        "--bloco", type=str, default=None,
        help="Bloco a treinar (A, B, C, D). Se omitido, usa BLOCO_ATIVO de app/main.py.",
    )
    parser.add_argument(
        "--datasource", type=str, default=None,
        choices=["csv", "database", "api"],
        help="Sobrescreve a fonte de dados ativa para este treino.",
    )
    args = parser.parse_args()

    # Permite override pontual sem mexer em app/config.py.
    # Útil em CI/CD: o pipeline define a fonte via variável e dispara
    # o treino sem editar arquivo nenhum.
    if args.datasource is not None:
        os.environ["_OVERRIDE_DATASOURCE"] = args.datasource
        # Reload do config — patch monkey-patch direto no módulo.
        import app.config as cfg
        cfg.DATASOURCE_KIND = args.datasource
        print(f"[train] datasource sobrescrito para: {args.datasource}")

    if args.bloco is None:
        # Lê BLOCO_ATIVO sem importar o app inteiro (evita carregar uvicorn etc.)
        from app.main import BLOCO_ATIVO
        bloco = BLOCO_ATIVO
    else:
        bloco = args.bloco

    print(f"=== Treinando Bloco {bloco} ===")
    metrics = _train_bloco(bloco)

    record = {
        "bloco": bloco,
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(record, indent=2, default=str))
    print(f"=== Métricas registradas em {METRICS_PATH} ===")


if __name__ == "__main__":
    main()
