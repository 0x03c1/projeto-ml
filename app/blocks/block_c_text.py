"""
Bloco C — Análise de Sentimento / Texto
========================================

OBJETIVO
--------
Classificar textos livres em rótulos semânticos (positivo, negativo,
neutro) ou em categorias específicas do domínio.

INTUIÇÃO DO BASELINE
--------------------
Um **classificador léxico**: contamos quantas palavras do texto
aparecem em uma lista pré-definida de termos positivos e negativos.

    score = (n_pos - n_neg) / (n_pos + n_neg)

A classe é decidida por limiares:
    score > 0.2  → positivo
    score < -0.2 → negativo
    senão        → neutro

Por que usar isso como BASELINE?
    1. Não precisa de dados rotulados para treinar — é "regra fixa".
    2. Roda sem GPU, sem download de modelo, em milissegundos.
    3. Estabelece uma linha de base que QUALQUER modelo refinado
       precisa superar para ser justificado.
    4. Em domínios técnicos (chamados de TI, observações médicas),
       léxicos genéricos têm performance notavelmente ruim — e isso
       é justamente o que motiva a Semana 4 (refinamento).

LIMITAÇÕES (que você DEVE discutir no relatório)
------------------------------------------------
- Ignora negação: "não gostei" tem 1 hit positivo, vira positivo.
- Ignora ironia/sarcasmo: "ótimo, mais um bug pra corrigir".
- Vocabulário técnico fora do léxico → sentimento neutro errado.

REFINAMENTO RECOMENDADO (Semana 4+)
-----------------------------------
Modelos pré-treinados em PT-BR — `pysentimiento` é a opção mais simples
(uma chamada de função e pronto). Internamente usa BERTimbau finetuned
em tweets brasileiros. Precisão sobe muito para textos curtos
informais (avaliações de e-commerce, comentários de rede social).

Para domínios MUITO específicos (jargão profissional, gírias regionais),
o modelo pré-treinado pode falhar — vale rotular ~200 exemplos do
domínio e treinar um classificador (LogisticRegression em cima dos
embeddings, por exemplo).
"""
from __future__ import annotations

from app.schemas import TextAnalysisResponse


# =============================================================
# Léxicos
# =============================================================
# Listas mínimas para começar. AMPLIE com base em uma amostra real
# do domínio: rode o baseline em ~50 textos, leia os erros, e
# adicione as palavras que estão faltando. Esse loop "olhar erro
# → ajustar léxico" é o que torna o baseline competitivo.
PALAVRAS_POSITIVAS: set[str] = {
    "ótimo", "otimo", "excelente", "bom", "boa", "maravilhoso", "perfeito",
    "adorei", "amei", "gostei", "recomendo", "incrível", "incrivel", "top",
    "satisfeito", "satisfeita", "feliz", "rápido", "rapido", "eficiente",
    "fantástico", "fantastico", "show",
}
PALAVRAS_NEGATIVAS: set[str] = {
    "ruim", "péssimo", "pessimo", "horrível", "horrivel", "terrível",
    "terrivel", "odiei", "detestei", "decepcionado", "decepcionada",
    "lento", "lenta", "demorado", "demorada", "caro", "cara", "frustrado",
    "frustrada", "problema", "defeito", "quebrado", "quebrada", "fraco",
    "fraca", "insatisfeito", "insatisfeita",
}


def analyze_baseline(text: str) -> TextAnalysisResponse:
    """Baseline contagem de polaridade. Útil para Semanas 1-3.

    Tokenização "burra" via `split()`. Para textos com pontuação
    grudada ("ótimo!"), removemos pontuação por token com `strip(".,!?;:")`.
    Em produção, considere `re.findall(r"\\w+", text.lower())` ou um
    tokenizador de verdade (nltk, spacy).
    """
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t.strip(".,!?;:") in PALAVRAS_POSITIVAS)
    neg = sum(1 for t in tokens if t.strip(".,!?;:") in PALAVRAS_NEGATIVAS)

    # Texto que não bate com nenhum termo do léxico: melhor admitir
    # incerteza ("neutro com score 0") do que chutar.
    if pos == 0 and neg == 0:
        return TextAnalysisResponse(
            sentimento="neutro",
            score=0.0,
            extras={"pos_hits": 0, "neg_hits": 0, "engine": "baseline_lexicon"},
        )

    total = pos + neg
    score = (pos - neg) / total

    # Limiares de 0.2 são heurísticos — escolhidos para evitar
    # classificar "positivo" um texto com 1 hit positivo e 0 hits
    # negativos (que daria score=1.0 e seria "muito positivo" sem
    # justificativa).
    if score > 0.2:
        sentimento = "positivo"
    elif score < -0.2:
        sentimento = "negativo"
    else:
        sentimento = "neutro"

    return TextAnalysisResponse(
        sentimento=sentimento,
        score=float(score),
        extras={"pos_hits": pos, "neg_hits": neg, "engine": "baseline_lexicon"},
    )


# =============================================================
# Modelo refinado — descomente após instalar `pysentimiento`
# =============================================================
# from pysentimiento import create_analyzer
# _analyzer = create_analyzer(task="sentiment", lang="pt")
#
# def analyze_transformer(text: str) -> TextAnalysisResponse:
#     out = _analyzer.predict(text)
#     mapping = {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}
#     score = float(out.probas[out.output])
#     return TextAnalysisResponse(
#         sentimento=mapping[out.output],
#         score=score,
#         extras={"engine": "pysentimiento", "probas": out.probas},
#     )


# =============================================================
# Função pública usada pelo endpoint
# =============================================================
def analyze(text: str) -> TextAnalysisResponse:
    """Roteia para o motor ativo.

    Mantenha o baseline funcional até o modelo refinado estar 100%
    testado. Trocar de motor é UMA linha — e os schemas Pydantic
    garantem que o frontend não percebe a diferença.
    """
    return analyze_baseline(text)
    # return analyze_transformer(text)
