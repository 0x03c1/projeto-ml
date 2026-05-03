"""
Configurações globais do microsserviço de ML.
==============================================

Este módulo concentra **constantes globais** usadas por todos os blocos
(A, B, C e D). A motivação para concentrar tudo aqui é dupla:

1. Reprodutibilidade — uma única semente aleatória `SEED` controla
   todos os experimentos. Alterar a semente em apenas um lugar muda o
   comportamento de todo o pipeline (split de treino/teste, embaralhamento
   de batches, geração de dados sintéticos, etc.).

2. Portabilidade — os caminhos `DATA_DIR` e `MODELS_DIR` são
   resolvidos a partir do diretório do projeto, e não do diretório de
   onde o script foi chamado. Isso evita o erro clássico de
   "FileNotFoundError" quando o serviço é executado de uma pasta diferente.

Não altere `SEED` no meio do projeto sem documentar a mudança no
relatório técnico. A reprodutibilidade dos experimentos depende disso.
"""
from pathlib import Path

# -----------------------------------------------------------------
# Reprodutibilidade
# -----------------------------------------------------------------
# Toda função que usa aleatoriedade DEVE receber esta semente. Em
# scikit-learn, isso significa passar `random_state=SEED`. Em numpy,
# `np.random.default_rng(SEED)`. Sem isso, dois treinos consecutivos
# produzem modelos diferentes e as métricas perdem comparabilidade.
SEED: int = 42

# -----------------------------------------------------------------
# Caminhos
# -----------------------------------------------------------------
# `BASE_DIR` aponta para a raiz do projeto (uma pasta acima de `app/`).
# `Path(__file__)` resolve o arquivo atual (`config.py`); o `.parent`
# duplo sobe dois níveis: app/config.py → app/ → raiz.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Onde ficam os datasets brutos. Não versionar arquivos pesados aqui;
# use `.gitignore` para excluir `data/*.csv` e versione apenas o
# `.gitkeep` para manter a pasta no Git.
DATA_DIR: Path = BASE_DIR / "data"

# Onde ficam os modelos serializados (.joblib, índices, etc.).
MODELS_DIR: Path = BASE_DIR / "models"

# Arquivo único onde gravamos as métricas do último treino. Este JSON
# é o "snapshot" que aparece no relatório técnico.
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

# Cria as pastas se ainda não existirem (evita erro no primeiro run).
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------
# Identificação do serviço
# -----------------------------------------------------------------
# Estes valores aparecem em `/health`. A aplicação web do projeto
# integrador pode usar `/health` como liveness probe antes de chamar
# `/predict`, `/recommend`, etc.
SERVICE_NAME: str = "ml_service"
SERVICE_VERSION: str = "0.2.0"

# -----------------------------------------------------------------
# Fonte de dados ativa
# -----------------------------------------------------------------
# Define de onde os blocos lêem seus dados. Possíveis valores:
#   - "csv"      → lê arquivos locais em `data/` (padrão; ideal para EDA)
#   - "database" → conecta no BD do projeto integrador (Grupo 1)
#   - "api"      → consome API REST pública (Grupo 2: IBGE, PRF, etc.)
#
# Cada bloco pode ainda sobrescrever esta escolha pontualmente, mas
# este é o default global.
DATASOURCE_KIND: str = "csv"
