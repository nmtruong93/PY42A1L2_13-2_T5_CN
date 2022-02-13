import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


class LogisticRegression:
    def __init__(self, epoch=200, learning_rate=0.0001, lamb=0.001):
        self.weight = None
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.lamb = lamb

    def __sigmoid(self, x_train): # e^(wx) / (1 + e^(wx + b))
        return np.exp(np.dot(self.weight, x_train)) / (1 + np.exp(np.dot(self.weight, x_train)))

    def fit(self, x_train, y_train):
        x_train = np.concatenate((x_train, np.ones((x_train.shape[0], 1))), axis=1)
        self.weight = np.zeros((1, x_train.shape[1]))
        errors = []
        for i in range(self.epoch):
            sig = self.__sigmoid(x_train.T)
            gradient = np.sum(np.dot((sig - y_train.T), x_train)) + self.lamb * self.weight
            self.weight = self.weight - self.learning_rate * gradient

            errors.append(np.sum((y_train.T - sig)**2))
            if self.epoch % 10 == 0:
                print(f"Error at epoch {i}/ {self.epoch}: {errors[-1]}")

        self.plot_error(errors)
        return self.weight

    def predict(self, x_test):
        x_test = np.concatenate((x_test, np.ones((x_test.shape[0], 1))), axis=1)

        mu = self.__sigmoid(x_test.T)

        return np.where(mu > 0.3, 1, 0)

    def predict_proba(self, x_test):
        x_test = np.concatenate((x_test, np.ones((x_test.shape[0], 1))), axis=1)

        return self.__sigmoid(x_test.T)

    def plot_error(self, errors):
        plt.plot(errors)
        plt.show()

    def _coefficient(self):
        return self.weight


if __name__ == '__main__':
    churn_data = pd.read_csv('ChurnData.csv')[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
                                               'callcard', 'wireless','churn']]
    churn_data = MinMaxScaler().fit_transform(churn_data)

    train, test = train_test_split(churn_data, test_size=0.2, random_state=4, shuffle=True)
    x_train, y_train = train[:, :-1], train[:, -1:].astype(int)
    x_test, y_test = test[:, :-1], test[:, -1:].astype(int)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)
    pred = logistic_regression.predict(x_test)
    print(confusion_matrix(y_test.T[0], pred[0], labels=[1,0]))
    # print(pred)

