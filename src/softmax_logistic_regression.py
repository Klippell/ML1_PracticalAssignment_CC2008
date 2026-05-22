# coding:utf-8
import logging
import autograd.numpy as np
from autograd import grad

try:
    from src.logistic_regression import BasicRegression
except ImportError:
    from logistic_regression import BasicRegression

# Configuração de semente para garantir reprodutibilidade
np.random.seed(42)

def softmax(x):
    """
    Função de ativação para classificação multinomial.
    Transforma scores (logits) em probabilidades que somam 1.
    """
    # Subtração do max para estabilidade numérica (evita exp(valor muito alto))
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def categorical_crossentropy(y_true, y_pred):
    """
    Calcula a perda (loss) para múltiplas classes.
    y_true deve estar em formato one-hot encoding.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class SoftmaxRegression(BasicRegression):
    
    def __init__(self, lr=0.001, penalty=None, C=0.01, tolerance=0.0001, 
                 max_iters=1000, weighted=False):  # <-- NOVO: parâmetro weighted
        """
        Softmax Regression com suporte opcional para Class Weighting.
        
        Parâmetros:
            weighted: bool, default=False
                Se True, aplica pesos por classe inversamente proporcionais 
                à frequência (Weighted Softmax). Útil em datasets 
                desbalanceados. Quando False, comporta-se como Softmax padrão.
        """
        # Reutiliza a inicialização da classe-mãe (BasicRegression)
        super().__init__(lr=lr, penalty=penalty, C=C, 
                         tolerance=tolerance, max_iters=max_iters)
        
        # NOVO: guardar configuração da ponderação
        self.weighted = weighted
        self.class_weights = None  # será calculado em fit() se weighted=True

    def init_cost(self):
        self.cost_func = categorical_crossentropy

    def fit(self, X, y):
        """
        Sobrescreve o fit da classe mãe para garantir o mapeamento de classes
        e a inicialização de pesos em matriz.
        """
        self._setup_input(X, y)
        
        # 1. Garante que X é 2D e float (evita erros matemáticos)
        self.X = np.atleast_2d(self.X).astype(float)
        
        # 2. MAPEAMENTO UNIVERSAL: Identifica classes e cria One-Hot Encoding
        self.classes_unique = np.unique(self.y)
        self.n_classes = len(self.classes_unique)
        
        # Cria um mapeamento interno (ex: "A" -> 0, "B" -> 1)
        label_to_idx = {val: i for i, val in enumerate(self.classes_unique)}
        y_indices = np.array([label_to_idx[val] for val in self.y])
        
        # O y usado no treino passa a ser a matriz de probabilidades reais (One-Hot)
        self.y_train_encoded = np.eye(self.n_classes)[y_indices]
        
        # 3. NOVO: Cálculo dos pesos por classe (só se weighted=True)
        # Fórmula: w_c = N / (K * n_c)  →  classes raras recebem peso maior
        if self.weighted:
            N = len(y_indices)
            K = self.n_classes
            class_counts = np.bincount(y_indices, minlength=K)
            # Adicionamos epsilon para evitar divisão por zero
            self.class_weights = N / (K * (class_counts + 1e-12))
            logging.info(f"Pesos das classes calculados (Inverse Frequency): {self.class_weights}")
        else:
            # Pesos todos iguais a 1 → matematicamente equivalente à Cross-Entropy padrão
            self.class_weights = np.ones(self.n_classes)
        
        # 4. Inicialização de pesos para MULTI-CLASSE
        self.n_samples, self.n_features = self.X.shape
        # O tamanho do theta deve ser (Features + 1) * Classes
        size = (self.n_features + 1) * self.n_classes
        self.theta = np.random.normal(size=size, scale=0.01)
        
        # 5. Adiciona bias e inicia o Gradient Descent
        self.X = self._add_intercept(self.X)
        self.init_cost()
        self._train()
    
    def _cost(self, X, y, theta):
        """
        Assinatura bate com a classe mãe (X, y, theta).
        Usa self.y_train_encoded (one-hot) e self.class_weights (NOVO).
        """
        n_features = X.shape[1]
        n_classes = theta.size // n_features
        w_matrix = theta.reshape(n_features, n_classes)
        logits = X.dot(w_matrix)
        # Softmax com estabilidade numérica
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Cross-entropy PONDERADA usando self.class_weights
        # (quando weighted=False, todos os pesos = 1 → cross-entropy padrão)
        cost = -np.mean(np.sum(self.class_weights * self.y_train_encoded * np.log(probs + 1e-12), axis=1))
        return cost

    def _loss(self, w):
        """
        Função de perda PONDERADA: aplica self.class_weights na cross-entropy.
        
        Loss padrão (weighted=False, todos os pesos = 1):
            L = -(1/N) Σ_i Σ_c y_ic * log(p_ic)
        
        Loss ponderada (weighted=True, pesos = N/(K*n_c)):
            L = -(1/N) Σ_i Σ_c w_c * y_ic * log(p_ic)
        """
        # Transforma o vetor de parâmetros 'w' de volta numa matriz (Features x Classes)
        w_matrix = w.reshape(self.n_features + 1, self.n_classes)
        
        # Cálculo dos Logits (Z = X.W) e Probabilidades (Softmax)
        z = np.dot(self.X, w_matrix)
        y_pred = softmax(z)
        
        # NOVO: aplica os pesos por classe na cross-entropy
        # Quando weighted=False, self.class_weights = [1, 1, ..., 1] → não tem efeito
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.mean(np.sum(self.class_weights * self.y_train_encoded * np.log(y_pred_clipped), axis=1))
        
        return self._add_penalty(loss, w)

    def predict_proba(self, X):
        """Calcula a matriz de probabilidades para cada classe."""
        X_array = np.atleast_2d(X).astype(float)
        X_intercept = self._add_intercept(X_array)
        w_matrix = self.theta.reshape(self.n_features + 1, self.n_classes)
        return softmax(np.dot(X_intercept, w_matrix))

    def predict(self, X):
        """Devolve o nome original da classe (String ou Int) mais provável."""
        probs = self.predict_proba(X)
        # Seleciona o índice com maior valor (axis=1 agora garantido pela matriz)
        indices = np.argmax(probs, axis=1)
        return self.classes_unique[indices]


# --- Exemplo de Uso ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Dataset sintético: 3 classes, 2 features
    X_train = np.array([[0.5, 0.2], [1.0, 1.5], [2.5, 0.5], [0.1, 0.1], [1.2, 1.2], [2.2, 0.8]])
    y_train = np.array([0, 1, 2, 0, 1, 2])

    # Modo baseline (sem ponderação)
    model_normal = SoftmaxRegression(lr=0.1, max_iters=500, weighted=False)
    model_normal.fit(X_train, y_train)
    print(f"[Baseline] Previsão para [2.0, 0.6]: {model_normal.predict([[2.0, 0.6]])}")
    
    # Modo proposta (com ponderação)
    model_weighted = SoftmaxRegression(lr=0.1, max_iters=500, weighted=True)
    model_weighted.fit(X_train, y_train)
    print(f"[Weighted] Previsão para [2.0, 0.6]: {model_weighted.predict([[2.0, 0.6]])}")