# Projeto Integrador Machine Learning (ML)

Código casca em FastAPI para o componente de Machine Learning do projeto integrador do 6º período. Cada equipe escolhe **um** dos quatro blocos disponíveis e o adapta ao domínio do seu projeto.

> **Importante:** este projeto é uma **casca** intencional. Há `TODO`s em pontos-chave que vocês devem implementar. As partes prontas (estrutura, endpoints, persistência, validação) servem para que vocês foquem no que importa: dados, modelo e métricas.

---

## Blocos disponíveis

| Bloco | Tema | Endpoint principal |
|-------|------|--------------------|
| A | Sistema de Recomendação | `GET /recommend` |
| B | Classificação / Predição | `POST /predict` |
| C | Análise de Sentimento / Texto | `POST /analyze` |
| D | Busca Semântica | `GET /search` |

Escolha **apenas um**. A escolha deve ser registrada no arquivo `BLOCO_ESCOLHIDO.md` (ver passo 4 abaixo).

---

## Como executar

### 1. Pré-requisitos
- Python 3.10 ou superior
- pip atualizado

### 2. Instalação
```bash
python -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Cada bloco tem dependências adicionais comentadas no `requirements.txt`. Descomente apenas as do seu bloco.

### 3. Execução
```bash
uvicorn app.main:app --reload --port 8000
```

A documentação interativa fica em `http://localhost:8000/docs`.

### 4. Registrar o bloco escolhido
Crie um arquivo `BLOCO_ESCOLHIDO.md` na raiz do projeto contendo:

```
Bloco escolhido: <A | B | C | D>
Justificativa de domínio: <2-3 frases ligando o bloco ao projeto da equipe>
Integrantes: <nomes>
```

---

## Estrutura

```
ml_service/
├── app/
│   ├── main.py              # FastAPI: rotas, CORS, healthcheck
│   ├── config.py            # configurações (semente, paths)
│   ├── schemas.py           # modelos Pydantic de request/response
│   └── blocks/
│       ├── block_a_recommender.py     # Bloco A
│       ├── block_b_classifier.py      # Bloco B
│       ├── block_c_text.py            # Bloco C
│       └── block_d_search.py          # Bloco D
├── data/                    # datasets (NÃO versionar arquivos pesados)
├── models/                  # modelos serializados
├── tests/                   # testes mínimos
├── train.py                 # script de treino (entry point)
├── requirements.txt
└── README.md
```

---

## Fluxo de trabalho recomendado

1. **Semana 1** — escolha do bloco, dataset definido, `BLOCO_ESCOLHIDO.md` criado.
2. **Semana 2** — EDA em notebook (pasta `notebooks/`, criada por vocês), salvar gráficos em `data/`.
3. **Semana 3** — implementar o **baseline** no bloco escolhido. Rodar `python train.py` e gerar `models/<bloco>.joblib` ou índice equivalente.
4. **Semana 4** — refinar o modelo, registrar métricas em `models/metrics.json`.
5. **Semana 5** — integrar com a aplicação web do projeto integrador. Habilitar CORS para o domínio da equipe.
6. **Semana 6** — polimento, README final, gravação de demo.

---

## Integração com a aplicação web

O microsserviço roda em uma porta separada (8000 por padrão). A aplicação web do projeto integrador deve consumir os endpoints via `fetch` (JS), `axios`, ou cliente HTTP equivalente da linguagem do backend de vocês.

Exemplo a partir de um frontend JS:

```javascript
const resp = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features: { idade: 35, renda: 4200 } })
});
const data = await resp.json();
console.log(data.prediction);
```

Em produção, a URL do microsserviço deve ser uma variável de ambiente do frontend.

---

## Regras importantes

- Modelo **baseline obrigatório** antes de qualquer modelo sofisticado.
- Semente aleatória **fixa** em todo treino (`SEED = 42` em `app/config.py`).
- Métricas registradas em `models/metrics.json` com data e versão.
- Commits frequentes — média mínima de 1 commit por integrante por semana.

---

## Suporte

Dúvidas técnicas: aulas de IA e horário de atendimento.
