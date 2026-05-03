"""
Camada de fontes de dados.

Esta pasta isola a *origem* dos dados do *uso* deles. Cada bloco (A, B,
C, D) chama uma função do tipo `load_*()` e não precisa saber se os
dados vieram de um CSV local, de um banco de dados PostgreSQL ou de
uma API REST pública.

Esse padrão se chama **Repository Pattern**. Em projetos de dados, ele
separa três responsabilidades que costumam virar bagunça quando
misturadas:

    +----------------+      +-------------+      +-----------+
    | Fonte de dados | ---> | Pré-process | ---> | Modelo ML |
    +----------------+      +-------------+      +-----------+

Aqui ficam apenas as funções da primeira caixa.
"""
