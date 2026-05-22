# Machine Learning I (CC2008) — Projeto Prático 2025/2026

**Avaliação e Adaptação de Algoritmos de Classificação**

Universidade do Porto — Faculdade de Ciências
Unidade Curricular: Machine Learning I (CC2008)
Turma: **PL3** | Grupo: **PL3_G1**
Docente: Rita Paula Almeida Ribeiro

## Equipa

| Número | Nome |
|--------|------|
| 202300276 | Guilherme Klippel |
| 202304280 | Daniel Tiago |
| 202512256 | Lucas Coelho |

## Resumo do Projeto

Este projeto investiga o desempenho da **Regressão Logística** em problemas de classificação **multiclasse** (Grupo de Datasets 3) e propõe duas adaptações sucessivas, avaliadas empiricamente sobre 45 datasets de benchmark.

A análise organiza-se em duas fases:

- **Fase 1 — Baseline.** Implementação da Logistic Regression de raiz (com `autograd` para diferenciação automática), construção de uma pipeline completa de pré-processamento (imputação, one-hot encoding, normalização Z-score) e avaliação via K-Fold Cross-Validation (k=5). A análise confirma a fragilidade estrutural da Logistic binária em contextos multiclasse (F1 macro médio ≈ 0.14).

- **Fase 2 — Proposta.** Extensão para **Softmax Regression** multiclasse, seguida de uma proposta original: **Weighted Softmax**, que aplica pesos por classe inversamente proporcionais à frequência (`w_c = N / (K · n_c)`) na cross-entropy. A comparação é validada estatisticamente com o teste de **Wilcoxon Signed-Rank** (Demšar, 2006), reconhecido como o protocolo adequado para comparar classificadores em múltiplos datasets.

## Principais Resultados

| Comparação | Métrica primária | Resultado | Significância |
|-----------|------------------|-----------|---------------|
| Logistic vs Softmax | F1 macro | Softmax vence (Δ = +0.479) | p ≈ 1.4 × 10⁻⁸ ✓ |
| Softmax vs Weighted Softmax (agregado) | F1 macro | Empate (Δ = −0.005) | p ≈ 0.10 ✗ |
| Softmax vs Weighted Softmax (Page-Blocks) | Recall macro | Weighted vence (Δ = +0.36) | Caso de estudo |

O Weighted Softmax produziu o trade-off teórico esperado (Recall ↑, Precision ↓) sem ganho líquido em F1 no agregado, mas demonstrou utilidade clara num estudo de caso com desbalanceamento severo (Page-Blocks).

## Estrutura do Repositório

```
Projeto/
├── final.ipynb                  Notebook principal (pipeline completa + análise)
├── slides.pdf                   Slides para apresentação (12 min)
├── requirements.txt             Dependências Python
├── README.md                    Este ficheiro
│
├── src/                         Implementação dos modelos
│   ├── logistic_regression.py   Logistic binária — Sigmoide + BCE
│   └── softmax_logistic_regression.py
│                                Softmax multiclasse + flag weighted=True
│
├── datasets/                    Benchmarks por tipo de desafio
│   ├── multiclass_classification/   45 datasets (foco do trabalho)
│   ├── class_imbalance/             50 datasets
│   └── noise_outliers/              50 datasets
│
├── resultados_iniciais/         Métricas brutas por dataset (Fase 1)
└── resultados_processados/      Resultados agregados para análise estatística
    ├── logistic/
    ├── softmax/
    └── weighted_softmax/
```

## Como Executar

**1. Instalar dependências**

```bash
pip install -r requirements.txt
```

**2. Abrir e executar o notebook**

```bash
jupyter notebook final.ipynb
```

Executa todas as células de cima para baixo. O notebook está organizado em 8 secções numeradas que correspondem ao fluxo da apresentação (pré-processamento → baseline → Softmax → testes estatísticos → Weighted Softmax → estudo de caso → conclusão).

**Nota:** os ficheiros das pastas `resultados_iniciais/` e `resultados_processados/` já incluem os resultados gerados pelas nossas execuções. Reexecutar o notebook irá sobrescrevê-los com novas execuções (resultados consistentes graças à seed fixa em `np.random.seed(42)`).

## Tecnologias Utilizadas

- **Python 3.12+**
- **autograd** — diferenciação automática para o gradient descent
- **NumPy / Pandas** — manipulação de dados
- **Matplotlib / Seaborn** — visualizações
- **scikit-learn** — apenas para métricas (`accuracy`, `precision/recall/f1 macro`) e divisão treino/teste, em conformidade com o enunciado
- **SciPy** — teste de Wilcoxon Signed-Rank

## Referências

- Implementação base inspirada em [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms) (referência indicada no enunciado), com modificações substanciais.
- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* Journal of Machine Learning Research, 7, 1–30. — protocolo do teste estatístico utilizado.