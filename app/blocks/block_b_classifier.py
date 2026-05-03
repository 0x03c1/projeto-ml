"""
Bloco B — Classificação / Predição
====================================

OBJETIVO
--------
Atribuir uma classe (ou valor numérico) a uma entidade do domínio,
a partir de um conjunto de características (features) numéricas e/ou
categóricas.

INTUIÇÃO DO BASELINE
--------------------
Usamos **Regressão Logística** dentro de um `Pipeline` do scikit-learn.
A regressão logística é um modelo linear que aprende um peso `w_j`
para cada feature `x_j` e calcula:

    z = w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n
    p(classe=1) = sigmoide(z) = 1 / (1 + e^(-z))

A previsão final é "classe 1" se p > 0.5, caso contrário "classe 0".

Por que esse modelo como BASELINE?
    1. Treina rápido (mesmo com 100k linhas).
    2. É interpretável: você consegue ler o peso de cada feature.
    3. Funciona surpreendentemente bem em muitos problemas reais.
    4. Se um modelo mais sofisticado (RandomForest, XGBoost) não
       superar a regressão logística, provavelmente o problema está
       nos DADOS, não no modelo.

Por que dentro de um `Pipeline`?
    Porque `StandardScaler` precisa ser aplicado antes da regressão
    logística (que é sensível à escala das features). O `Pipeline`
    garante que o scaling é "aprendido" no fit (média/desvio do treino)
    e aplicado igualzinho no test e na predição online. Sem isso,
    é fácil cometer **data leakage** (vazar estatísticas do test
    para o treino) e ter métricas otimistas demais.

REFINAMENTOS POSSÍVEIS (Semana 4+)
----------------------------------
- RandomForestClassifier (não-linear, robusto a outliers).
- GradientBoostingClassifier (geralmente o melhor "off-the-shelf").
- StratifiedKFold para validação cruzada (mais confiável que um
  único train_test_split).
- GridSearchCV para tuning de hiperparâmetros.
- Análise de erro: matriz de confusão + revisar os falsos positivos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import METRICS_PATH, MODELS_DIR, SEED
from app.datasources.base import get_datasource
from app.schemas import PredictResponse

MODEL_PATH: Path = MODELS_DIR / "block_b_classifier.joblib"

# =============================================================
# Configuração — ALTERAR conforme o domínio da equipe
# =============================================================
# Estas duas constantes são o ÚNICO ponto de adaptação que a equipe
# precisa mexer (assumindo que a fonte de dados já devolve as colunas
# certas). Mantenha os nomes consistentes entre treino e predição
# online — qualquer divergência aqui gera bug silencioso.
TARGET_COLUMN: str = "target"
FEATURE_COLUMNS: list[str] = ["feature_1", "feature_2", "feature_3"]


# =============================================================
# Treino
# =============================================================
def train() -> dict:
    """Treina o baseline e persiste em MODEL_PATH."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = get_datasource().fetch_dataset()
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset: {missing}. "
            f"Verifique FEATURE_COLUMNS/TARGET_COLUMN ou a fonte de dados."
        )

    # ----------------------------------------------------------------
    # 1. Separação X / y
    # ----------------------------------------------------------------
    # X é a matriz de features (n_amostras × n_features).
    # y é o vetor de targets (n_amostras,).
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    # ----------------------------------------------------------------
    # 2. Holdout: 80% treino / 20% teste, ESTRATIFICADO
    # ----------------------------------------------------------------
    # `stratify=y` mantém a proporção de classes igual em treino e teste.
    # Isso é crítico em datasets desbalanceados — sem isso, é possível
    # cair num split em que a classe minoritária some quase toda do teste,
    # produzindo métricas absurdas.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ----------------------------------------------------------------
    # 3. Pipeline: scaling + modelo
    # ----------------------------------------------------------------
    # `StandardScaler`: subtrai a média e divide pelo desvio. Faz com
    # que cada feature tenha média 0 e variância 1.
    # `LogisticRegression(max_iter=1000)`: 100 iterações (default) às
    # vezes não basta para convergir em datasets pequenos.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipeline.fit(X_train, y_train)

    # ----------------------------------------------------------------
    # 4. Avaliação no holdout
    # ----------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ----------------------------------------------------------------
    # 5. Persistência (modelo + metadados)
    # ----------------------------------------------------------------
    # Salvamos a lista de colunas DENTRO do bundle. Assim, a função
    # `predict` em produção lê os nomes esperados de lá, evitando
    # divergência se alguém mexer em `FEATURE_COLUMNS` depois.
    bundle = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "classes": [str(c) for c in pipeline.classes_],
        "model_type": "logistic_regression_baseline",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "model_type": bundle["model_type"],
        "classes": bundle["classes"],
        "report": report,
    }
    print(f"[Bloco B] Modelo salvo em {MODEL_PATH}")
    print(f"[Bloco B] Accuracy: {acc:.4f}")

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


# =============================================================
# Inferência
# =============================================================
_cache: dict | None = None


def _load_bundle() -> dict:
    global _cache
    if _cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        _cache = joblib.load(MODEL_PATH)
    return _cache


def predict(features: dict[str, Any]) -> PredictResponse:
    """Prediz classe e probabilidades a partir do dict de features.

    Os nomes das chaves devem bater com `feature_columns` salvas no bundle.
    Ordem das colunas é preservada para evitar trocar features na hora errada.
    """
    bundle = _load_bundle()
    cols: list[str] = bundle["feature_columns"]

    # Validação de presença: nunca deixe o sklearn errar com
    # KeyError críptico. Erro explícito > erro silencioso.
    missing = [c for c in cols if c not in features]
    if missing:
        raise KeyError(", ".join(missing))

    # `[[...]]` cria matriz (1, n_features). O modelo espera 2D
    # mesmo para um único exemplo.
    X = np.array([[features[c] for c in cols]], dtype=float)
    pipeline = bundle["pipeline"]

    pred = pipeline.predict(X)[0]
    proba_dict: dict[str, float] | None = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        proba_dict = {cls: float(p) for cls, p in zip(bundle["classes"], proba)}

    return PredictResponse(prediction=str(pred), proba=proba_dict)
