# Machine Learning I (CC2008) — Projeto Prático 2025/2026

**Avaliação e Adaptação de Algoritmos de Classificação**
*Aplicação ao problema de classificação multiclasse (Grupo de Datasets 3)*

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

---

## 1. Síntese do Trabalho

Este projeto implementa de raiz três algoritmos de classificação encadeados — **Logistic Regression**, **Softmax Regression** e **Weighted Softmax Regression** — e estuda empiricamente o seu comportamento em **45 datasets multiclasse** de benchmark. A proposta original do grupo é a variante Weighted Softmax, que aplica pesos por classe inversamente proporcionais à frequência na função de perda, com o objectivo de mitigar o sub-aprendizado de classes minoritárias em problemas multiclasse desbalanceados.

A análise está estruturada em duas fases (alinhadas com o enunciado) e adopta um protocolo experimental rigoroso: K-Fold Cross-Validation com k=5, prevenção activa de *data leakage*, avaliação por quatro métricas (Accuracy, Precision macro, Recall macro, F1 macro), e validação estatística por teste de **Wilcoxon Signed-Rank** segundo o protocolo de Demšar (2006).

### Resultados Sintéticos

| Comparação | Métrica | Δ médio | p-value | Conclusão |
|-----------|---------|---------|---------|-----------|
| Softmax vs Logistic | F1 macro | +0.479 | 1.39 × 10⁻⁸ | **Softmax vence** com significância massiva |
| Softmax vs Logistic | Accuracy | +0.433 | 1.59 × 10⁻⁸ | Softmax vence |
| Weighted vs Softmax (agregado) | F1 macro | −0.005 | 0.099 | Empate estatístico |
| Weighted vs Softmax (agregado) | Recall macro | +0.023 | 0.227 | Empate estatístico |
| Weighted vs Softmax (Page-Blocks) | Recall macro | +0.359 | (caso de estudo) | Weighted vence |

O Weighted Softmax produziu exactamente o trade-off teórico esperado (Recall ↑, Precision ↓, Accuracy ↓) sem ganho líquido em F1 no agregado — mas o estudo de caso no dataset Page-Blocks demonstra a sua aplicabilidade em cenários de desbalanceamento severo (classe minoritária com 28 amostras contra 4913 da maioritária).

---

## 2. Algoritmos Implementados

Todas as implementações são feitas de raiz com `autograd.numpy` para diferenciação automática do gradiente — sem recurso a `scikit-learn` para a parte de modelagem (scikit-learn é usado apenas para métricas e divisão treino/teste, em conformidade com o enunciado).

### 2.1 Logistic Regression (baseline)

Classificador binário definido pela função sigmoide:

$$P(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^\top \mathbf{x} + b$$

Implementação em `src/logistic_regression.py`. A sigmoide é reescrita como `0.5 * (tanh(0.5x) + 1)` — matematicamente idêntica à fórmula clássica, mas numericamente estável para valores extremos de `z`. A função de perda é a **Binary Cross-Entropy** com clipping de probabilidade por epsilon (`1e-15`) para evitar `log(0)`.

A classe-base `BasicRegression` encapsula o gradient descent, regularização L1/L2 opcional, adição do intercept e critério de paragem por tolerância — sendo herdada pela Softmax Regression.

### 2.2 Softmax Regression (extensão para multiclasse)

Para K classes, o modelo calcula um vector de logits e converte-o em distribuição de probabilidade:

$$P(y=c \mid x) = \frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}}$$

Implementação em `src/softmax_logistic_regression.py`. A função `softmax()` aplica subtração do máximo (`exp(z - max(z))`) para estabilidade numérica — evitando overflow do `exp` em logits grandes. A perda é a **Categorical Cross-Entropy** sobre one-hot encoding das classes:

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} y_{ic} \log(p_{ic})$$

O `fit()` faz mapeamento universal de classes (suporta tanto labels inteiros como strings), e a matriz de pesos tem forma `(n_features+1) × K`.

### 2.3 Weighted Softmax Regression (proposta original)

Ativada pela flag `weighted=True` no construtor. A modificação substitui a Categorical Cross-Entropy padrão por uma versão ponderada:

$$L_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} w_c \cdot y_{ic} \log(p_{ic})$$

onde os pesos `w_c` são calculados a partir das frequências observadas no conjunto de treino:

$$w_c = \frac{N}{K \cdot n_c}$$

com `N` = número total de amostras, `K` = número de classes, `n_c` = número de amostras da classe `c`. Esta fórmula garante que classes raras recebem peso maior no gradiente (forçando o modelo a "prestar atenção" a elas), enquanto a média ponderada total se mantém igual a 1 (não distorce a escala global da loss).

**Motivação teórica:** durante a análise dos resultados da Softmax padrão, observou-se um gap sistemático de ~0.12 pontos entre Accuracy (0.745) e F1 macro (0.620). Este gap é a assinatura clássica de desbalanceamento de classes — a Accuracy beneficia das classes maioritárias enquanto o F1 macro, que pondera classes equitativamente, expõe o sub-aprendizado das minoritárias. A ponderação inversa por frequência é a contra-medida directa: alinha o sinal do gradiente com a importância das classes em vez da sua representatividade amostral.

---

## 3. Pipeline de Pré-Processamento

Implementada no notebook `final.ipynb`, secção 2:

1. **Imputação de valores nulos:** mediana para variáveis numéricas (robusta a outliers), moda para categóricas.
2. **One-Hot Encoding:** conversão de variáveis qualitativas em binárias, com remoção da primeira coluna de cada grupo (`drop_first=True`) para evitar multicolinearidade perfeita.
3. **Normalização Z-Score:** padronização para média 0 e desvio padrão 1, garantindo escalas comparáveis entre features.
4. **Prevenção de data leakage:** os parâmetros estatísticos (média, desvio, mediana, moda) são calculados **exclusivamente no fold de treino** e depois aplicados ao fold de teste. Nenhuma estatística é calculada sobre o dataset completo.
5. **K-Fold Cross-Validation:** k=5 para todos os experimentos, garantindo que cada amostra é usada como teste exactamente uma vez.

---

## 4. Protocolo Experimental

- **Datasets:** 45 datasets multiclasse de benchmark (em `datasets/multiclass_classification/`). Os outros dois grupos (`class_imbalance/`, `noise_outliers/`) foram usados apenas no diagnóstico inicial (slide 5) para fundamentar a escolha do Grupo 3.
- **Métricas:** Accuracy, Precision macro, Recall macro, F1 macro — calculadas via `sklearn.metrics`. A média macro é deliberada por dar peso igual a todas as classes (revela problemas em minoritárias).
- **Validação:** K-Fold com k=5, semente fixa (`np.random.seed(42)`).
- **Hiperparâmetros:** `lr=0.001`, `max_iters=1000`, `tolerance=1e-4`, `penalty=None` (regularização desactivada nesta análise).
- **Teste estatístico:** **Wilcoxon Signed-Rank** (`scipy.stats.wilcoxon`), recomendado por Demšar (2006) para comparação de classificadores em múltiplos datasets por ser:
  - Não-paramétrico (não assume normalidade das diferenças).
  - Pareado (compara dataset por dataset, não médias globais).
  - Robusto a outliers.
- **Nível de significância:** α = 0.05.

---

## 5. Estrutura do Repositório

```
Projeto/
├── final.ipynb                  Notebook principal (8 secções, top-to-bottom executável)
├── slides.pdf                   Apresentação para 12 min (14 slides)
├── requirements.txt             Dependências Python com versões mínimas
├── README.md                    Este ficheiro
│
├── src/                         Implementação dos modelos (de raiz)
│   ├── __init__.py
│   ├── logistic_regression.py
│   │     - BasicRegression: classe-base com gradient descent, L1/L2,
│   │       intercept e tolerância
│   │     - LogisticRegression(BasicRegression): sigmoide + BCE
│   └── softmax_logistic_regression.py
│         - SoftmaxRegression(BasicRegression): softmax + CCE
│         - Flag weighted=True activa a variante ponderada
│         - One-hot encoding interno, suporta labels int ou string
│
├── datasets/                    Benchmarks (organizados por característica)
│   ├── multiclass_classification/   45 datasets — foco do trabalho
│   ├── class_imbalance/             50 datasets — usados no diagnóstico inicial
│   └── noise_outliers/              50 datasets — usados no diagnóstico inicial
│
├── resultados_iniciais/         Métricas brutas por dataset (Fase 1)
│   ├── evaluation_baseline_grupo1_noise.csv
│   ├── evaluation_baseline_grupo2_imbalance.csv
│   └── evaluation_baseline_grupo3_multiclass.csv
│
└── resultados_processados/      Resultados agregados para análise estatística
    ├── logistic/                Logistic em todos os grupos
    ├── softmax/                 Softmax no Grupo 3 multiclasse
    └── weighted_softmax/        Weighted Softmax no Grupo 3 multiclasse
```

---

## 6. Como Executar

**Pré-requisitos:** Python 3.10+

**Passo 1 — Instalar dependências:**

```bash
pip install -r requirements.txt
```

**Passo 2 — Abrir o notebook:**

```bash
jupyter notebook final.ipynb
```

**Passo 3 — Executar de cima para baixo.** O notebook está organizado em 8 secções numeradas que correspondem ao fluxo da apresentação:

1. Introdução e contexto.
2. Pré-processamento e pipeline.
3. Resultados Logistic baseline.
4. Comparação Logistic vs Softmax.
5. Teste de hipótese #1 (Wilcoxon).
6. Motivação da proposta (gap Acc−F1).
7. Estudo de caso (Page-Blocks).
8. Conclusão geral.

Os ficheiros em `resultados_iniciais/` e `resultados_processados/` já contêm os resultados das nossas execuções (graças à seed fixa, são reprodutíveis bit-a-bit). Reexecutar o notebook irá sobrescrevê-los com idênticos valores.

**Nota sobre tempos de execução:** o treino completo nos 45 datasets multiclasse demora aproximadamente 8–12 minutos numa máquina recente (depende sobretudo do número de iterações até convergência em cada dataset).

---

## 7. Dependências

| Pacote | Versão mínima | Uso |
|--------|---------------|-----|
| `numpy` | 1.24.0 | Operações vectoriais |
| `pandas` | 2.0.0 | Manipulação de datasets (CSV → DataFrame) |
| `autograd` | 1.6.2 | Diferenciação automática do gradiente |
| `scikit-learn` | 1.2.0 | Apenas para `metrics` e `train_test_split` |
| `scipy` | 1.10.0 | Teste de Wilcoxon Signed-Rank |
| `matplotlib` | 3.7.0 | Gráficos |
| `seaborn` | 0.12.0 | Histogramas e visualizações comparativas |
| `notebook` / `ipykernel` | 7.0 / 6.0 | Execução do Jupyter |

---

## 8. Referências

- **Implementação base:** [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms) — repositório indicado no enunciado como referência open-source. A classe `BasicRegression` e a estrutura geral do gradient descent foram inspiradas neste código, com modificações substanciais para suportar multiclasse e ponderação por classe.
- **Demšar, J. (2006).** *Statistical Comparisons of Classifiers over Multiple Data Sets.* Journal of Machine Learning Research, 7, 1–30. — protocolo do teste estatístico utilizado para comparar classificadores em múltiplos datasets.
- **King, G. & Zeng, L. (2001).** *Logistic Regression in Rare Events Data.* — fundamentação teórica para o uso de ponderação por classe em problemas desbalanceados.

---

## 9. Notas sobre Limitações e Trabalho Futuro

A análise agregada nos 45 datasets indica que o Weighted Softmax não produz ganho líquido em F1 macro — apesar de produzir o trade-off teórico esperado entre Recall e Precision. Hipotetizamos que a fórmula `w_c = N / (K · n_c)` é demasiado agressiva em datasets onde o desbalanceamento é moderado, levando o modelo a "exagerar" na compensação. Direcções futuras incluem:

- **Suavização dos pesos:** fórmulas alternativas como `w_c = sqrt(N / (K · n_c))` ou `w_c = log(N / n_c)`, que penalizam menos o desbalanceamento moderado.
- **Técnicas a nível dos dados:** combinação com SMOTE ou *random oversampling* das classes minoritárias antes do treino.
- **Métricas adaptativas:** uso de F-beta com β > 1 quando o problema requer prioritização explícita do Recall.