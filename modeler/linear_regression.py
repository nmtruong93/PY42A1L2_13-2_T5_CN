import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader.housing_price_loader import load_boston_data_v2
warnings.filterwarnings('ignore')


def modeling(X_train, X_test, y_train, y_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)  # training model

    prediction = linear_model.predict(X_test)  # Generalization.

    return linear_model, pd.DataFrame({'actual': y_test, 'prediction': prediction})


if __name__ == '__main__':

    data_df = load_boston_data_v2()
    X, y = data_df.iloc[:, :-1], data_df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=4)

    linear_model, results = modeling(X_train, X_test, y_train, y_test)
    print(f"MSE: {mean_squared_error(results.actual, results.prediction)}")
    print(f"MAE: {mean_absolute_error(results.actual, results.prediction)}")
    print(f"R2-Score: {r2_score(results.actual, results.prediction)}")

    # results['feature'] = X.iloc[:, 0]
    # plt.scatter(results.feature, results.actual, label='Actual')
    # plt.scatter(results.feature, results.prediction, label='Prediction')
    # plt.legend()
    # plt.show()
