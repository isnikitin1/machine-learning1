from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, delta: float = 1):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = LossFunction(loss_function)
        self.delta = delta
        
    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))
    
    def calc_loss(self, x, y):
        """
            Calculate loss for x and y with our weights
            :param x: features array
            :param y: targets array
            :return: loss: float
        """
        self.x = x.copy()
        self.y = y.copy()
        if self.loss_function == LossFunction.MSE:
            self.loss = 1 / self.y.shape[0] * (self.y - self.x @ self.w).T @ (self.y - self.x @ self.w)
        if self.loss_function == LossFunction.LogCosh:
            self.loss = np.sum(np.log(np.cosh(self.x @ self.w - self.y))) / self.y.shape[0]
        if self.loss_function == LossFunction.MAE:
            self.loss = np.sum(np.abs(self.x @ self.w - self.y)) / self.y.shape[0]
        if self.loss_function == LossFunction.Huber:
            if np.linalg.norm(self.y - self.x @ self.w) <= self.delta:
                self.loss = 1 / (2 * self.y.shape[0]) * (self.y - self.x @ self.w).T @ (self.y - self.x @ self.w)
            else:
                self.loss = - np.sum(self.delta * (np.abs(self.x @ self.w - self.y) - 1 / 2 * self.delta)) / self.y.shape[0]
        return self.loss
                
    def predict(self, X):
        """
            Calculate predictions for x
            :param x: features array
            :return: prediction: np.ndarray
        """
        self.X = X.copy()
        self.y_pred = self.X @ self.w
        return self.y_pred

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
    

class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """
    
    def calc_gradient(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        if self.loss_function == LossFunction.MSE:
            return -2 / y.shape[0] * self.x.T @ (self.y - (self.x @ self.w))
        if self.loss_function == LossFunction.LogCosh:
            return self.x.T @ np.tanh(self.x @ self.w - self.y) / self.y.shape[0]
        if self.loss_function == LossFunction.MAE:
            return self.x.T @ np.sign(self.x @ self.w - self.y) / self.y.shape[0]
        if self.loss_function == LossFunction.Huber:
            if np.linalg.norm(self.y - self.x @ self.w) <= self.delta:
                return 1 / self.y.shape[0] * self.x.T @ (self.y - (self.x @ self.w))
            else:
                return self.delta * self.x.T @ np.sign(x @ self.w - self.y) / self.y.shape[0]
        
    def update_weights(self, gradient):
        """
            :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.gradient = gradient.copy()
        self.diff = - self.lr() * gradient
        self.w += self.diff
        return self.diff


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

<<<<<<< Updated upstream
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
=======
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50, loss_function: LossFunction = LossFunction.MSE):
>>>>>>> Stashed changes
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size
        
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        index = np.random.randint(0, y.shape[0], self.batch_size)
        self.x = x.copy()[index]
        self.y = y.copy()[index]
        if self.loss_function == LossFunction.MSE:
            return -2 / self.y.shape[0] * self.x.T @ (self.y - (self.x @ self.w))
        if self.loss_function == LossFunction.LogCosh:
            return self.x.T @ np.tanh(self.x @ self.w - self.y) / self.y.shape[0]
        if self.loss_function == LossFunction.MAE:
            return self.x.T @ np.sign(self.x @ self.w - self.y) / self.y.shape[0]
        if self.loss_function == LossFunction.Huber:
            if np.linalg.norm(self.y - self.x @ self.w) <= self.delta:
                return 1 / self.y.shape[0] * self.x.T @ (self.y - (self.x @ self.w))
            else:
                return self.delta * self.x.T @ np.sign(self.x @ self.w - self.y) / self.y.shape[0]
    

class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9
        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.gradient = gradient.copy()
        if self.lr.iteration == 0:
            self.h_k = self.h
        self.h_k = + self.alpha * self.h_k + self.lr() * self.gradient
        self.w = self.w - self.h_k
        return -self.h_k

class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.n_k: float = 0.01
        
        self.iteration: int = 0
    
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.gradient = gradient
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * self.gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(self.gradient, 2)
        self.iteration += 1
        self.diff = -self.lr() * self.m / (1 - self.beta_1 ** self.iteration) / (np.sqrt(self.v / (1 - self.beta_2 ** self.iteration)) + self.eps)
        self.w += self.diff
        return self.diff

class Nadam(Adam):
        
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.gradient = gradient
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * self.gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(self.gradient, 2)
        self.iteration += 1
        self.diff = (-self.lr() / (np.sqrt(self.v / (1 - np.power(self.beta_2, self.iteration))) + self.eps)
            * (self.m * self.beta_1 / (1 - np.power(self.beta_1, self.iteration)) + (1 - self.beta_1) / (
                1 - np.power(self.beta_1, self.iteration)) * self.gradient))
        self.w += self.diff
        return self.diff

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w
        l2_gradient[-1] = 0
        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'nadam': Nadam
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
