from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import streamlit as st


def load_boston_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    return None


def load_boston_data_v2():
    data = load_boston()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['PRICE'] = data.target

    return df


if __name__ == '__main__':
    # load_boston_data()
    df = load_boston_data_v2()
    df
