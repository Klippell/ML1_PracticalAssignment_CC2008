# Planeamento Técnico - Trabalho Prático ML1 (CC2008)

Este documento detalha o estado atual do projeto, os erros identificados na Baseline e as tarefas pendentes para o Checkpoint de Abril e a Entrega Final.

## 1. Erros Identificados na Fase 1 (Baseline)

Durante os testes de benchmarking, foram detetados os seguintes problemas que impedem a execução correta em todos os datasets:

* **Incompatibilidade de Tipos (Strings):** O algoritmo falha ao processar colunas categóricas.
  * *Solução:* Aplicar One-Hot Encoding nas features e Label Encoding no alvo.
* **Instabilidade Numérica (Overflow):** Erros de cálculo exponencial em escalas de dados díspares.
  * *Solução:* Implementar a normalização de dados via StandardScaler antes do treino.
* **Valores Ausentes (NaN):** O modelo interrompe a execução ao encontrar dados em falta.
  * *Solução:* Aplicar imputação estatística (média/moda) em todos os sub-datasets.

## 2. Tarefas Pendentes - Checkpoint (Abril)

* [ ] Analisar os ficheiros CSV de resultados consolidados para identificar as maiores fraquezas do modelo base.
* [ ] Selecionar formalmente um dos três grupos de desafio para a Fase 2 (Ruído, Desequilíbrio ou Multiclasse).
* [ ] Redigir a fundamentação teórica da Regressão Logística no Jupyter Notebook.

## 3. Tarefas Pendentes - Entrega Final (Maio)

* [ ] Desenvolver e implementar a variante robusta (Modificação do Algoritmo).
* [ ] Realizar o estudo empírico comparativo entre a Baseline e a versão proposta.
* [ ] Elaborar o relatório final e a estrutura de apresentação para discussão.
