from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        np.arange(x.shape[0])
        ind_train, _, _, _ = train_test_split(np.arange(x.shape[0]), 
                                                            np.arange(x.shape[0]), 
                                                            test_size = 1 - self.subsample, 
                                                            random_state = 51261)
        ind_cur = np.random.choice(ind_train.shape[0], ind_train.shape[0])
        model = self.base_model_class(**self.base_model_params).fit(x[ind_cur], 
                                            - self.loss_derivative(y[ind_cur], predictions[ind_cur]))
        
        self.gammas.append(self.find_optimal_gamma(y, predictions, model.predict(x)))
        self.models.append(model)
        
    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0]).reshape(1, y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0]).reshape(1, y_valid.shape[0])
        train_losses = [0]
        valid_losses = [0]
        i = 0
        for n in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions[-1])
            train_predictions = np.vstack((train_predictions, train_predictions[-1] +
                                          self.learning_rate * self.gammas[-1] * 
                                          self.models[-1].predict(x_train)))
            valid_predictions = np.vstack((valid_predictions, valid_predictions[-1] +
                                          self.learning_rate * self.gammas[-1] * 
                                          self.models[-1].predict(x_valid)))
            train_losses.append(self.loss_fn(y_train, train_predictions[-1]))
            valid_losses.append(self.loss_fn(y_valid, valid_predictions[-1]))
            if self.early_stopping_rounds is not None:
                if valid_losses[-1] > valid_losses[-2]:
                    i+= 1
                else:
                    i = 0
                if i >= self.early_stopping_rounds: 
                    break

        if self.plot:
            plt.plot(np.arange(len(valid_losses))[1:], valid_losses[1:])
            plt.xlabel('base model number')
            plt.ylabel('valid loss')
            plt.show()
        return self
    
    def predict_proba(self, x):
        prediction = 0
        for gamma, model in zip(self.gammas, self.models):
            prediction += self.learning_rate * gamma * model.predict(x)
        return np.array([1 - self.sigmoid(prediction), self.sigmoid(prediction)]).T
            

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        importances = np.zeros((len(self.models), self.models[0].feature_importances_.shape[0]))
        for i in range(len(self.models)):
            importances[i] = self.models[i].feature_importances_
        imp = np.mean(importances, axis = 0) / np.sum(np.mean(importances, axis = 0))
        return imp