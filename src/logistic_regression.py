# coding:utf-8
import logging
import autograd.numpy as np
from autograd import grad

# Configuração de semente para garantir reprodutibilidade nos testes
np.random.seed(1000)

def binary_crossentropy(y_true, y_pred):
    """
    Calcula o erro logístico (Log Loss).
    Usamos autograd.numpy (np) para que o autograd consiga calcular a derivada depois.
    """
    # Adicionamos um 'epsilon' minúsculo para evitar calcular log(0), que daria erro (NaN)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class BasicRegression:
    def __init__(self, lr=0.001, penalty=None, C=0.01, tolerance=0.0001, max_iters=1000):
        """
        Classe base para otimização usando Gradient Descent.
        """
        self.lr = lr                # Taxa de aprendizagem (o tamanho do 'passo' no gradient descent)
        self.penalty = penalty      # Regularização ('l1', 'l2' ou None)
        self.C = C                  # Coeficiente de regularização
        self.tolerance = tolerance  # Critério de paragem (se o erro mudar muito pouco, paramos)
        self.max_iters = max_iters  # Número máximo de iterações
        self.errors = []            # Histórico de erros para plotar gráficos depois
        self.theta = []             # Os pesos (parâmetros) do nosso modelo
        self.n_samples = None
        self.n_features = None
        self.cost_func = None

    def _setup_input(self, X, y=None):
        """Garante que os inputs são arrays do numpy."""
        self.X = np.array(X)
        if y is not None:
            self.y = np.array(y)

    def _loss(self, w):
        raise NotImplementedError()

    def init_cost(self):
        raise NotImplementedError()

    def _add_penalty(self, loss, w):
        """Aplica regularização (penalização) para evitar Overfitting."""
        if self.penalty == "l1":
            loss += self.C * np.abs(w[1:]).sum()
        elif self.penalty == "l2":
            loss += (0.5 * self.C) * (w[1:] ** 2).sum()
        return loss

    def _cost(self, X, y, theta):
        """Calcula o custo atual da previsão."""
        prediction = X.dot(theta)
        error = self.cost_func(y, prediction)
        return error

    def fit(self, X, y):
        """Treina o modelo com os dados X e rótulos y."""
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = self.X.shape

        # Inicializa os pesos (theta) aleatoriamente, incluindo o termo de bias
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)

        # Adiciona uma coluna de 1s para representar o Intercept (Bias)
        self.X = self._add_intercept(self.X)

        self._train()

    @staticmethod
    def _add_intercept(X):
        """Adiciona uma coluna de 1s na matriz X para o cálculo do Bias."""
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def _train(self):
        """Inicia o processo de Gradient Descent."""
        self.theta, self.errors = self._gradient_descent()
        logging.info(" Pesos Finais (Theta): %s" % self.theta.flatten())

    def _gradient_descent(self):
        """O coração do algoritmo: otimização dos pesos."""
        theta = self.theta
        errors = [self._cost(self.X, self.y, theta)]
        
        # A MAGIA ACONTECE AQUI: autograd calcula a derivada da função de loss automaticamente!
        cost_d = grad(self._loss)
        
        for i in range(1, self.max_iters + 1):
            # Calcula o gradiente e atualiza os pesos
            delta = cost_d(theta)
            theta -= self.lr * delta

            errors.append(self._cost(self.X, self.y, theta))
            
            # Verifica se o algoritmo convergiu (se parou de aprender coisas novas)
            error_diff = np.linalg.norm(errors[i - 1] - errors[i])
            if error_diff < self.tolerance:
                logging.info(f"Convergência alcançada na iteração {i}.")
                break
                
        return theta, errors


class LogisticRegression(BasicRegression):
    """
    Classificador de Regressão Logística Binária.
    """

    def init_cost(self):
        # Define a função de custo como Cross Entropy
        self.cost_func = binary_crossentropy

    def _loss(self, w):
        # Função que o autograd vai derivar. Passa os dados pela Sigmoide antes de calcular o erro.
        loss = self.cost_func(self.y, self.sigmoid(np.dot(self.X, w)))
        return self._add_penalty(loss, w)

    @staticmethod
    def sigmoid(x):
        """Função de ativação Sigmoide (transforma qualquer número num valor entre 0 e 1)."""
        return 0.5 * (np.tanh(0.5 * x) + 1)

    def predict_proba(self, X):
        """Retorna as probabilidades (ex: 0.85 de ser da classe 1)."""
        X_numpy = np.array(X)
        X_intercept = self._add_intercept(X_numpy)
        return self.sigmoid(X_intercept.dot(self.theta))
        
    def predict(self, X, threshold=0.5):
        """Retorna a classe final (0 ou 1) baseada num limiar."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)