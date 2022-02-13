import _pickle
import json

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    with open('../data/bow_train.p', 'rb') as file:
        data_train = _pickle.load(file)

    with open('../data/bow_test.p', 'rb') as file:
        data_test = _pickle.load(file)

    with open('../data/dictionary.json', 'r') as file:
        dictionary = json.load(file)

    x_train = np.asarray(data_train['vector'])
    print(x_train.shape)
    y_train = np.asarray(data_train['label'])

    x_test = np.asarray(data_test['vector'])
    y_test = np.asarray(data_test['label'])

    inputs = keras.Input(shape=(len(dictionary),))
    dense = layers.Dense(32, activation="relu")
    x = dense(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(5)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model")

    print(model.summary())

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=32, epochs=5)

    with open('model.p', 'wb') as file:
        _pickle.dump(model, file)

    print(model.evaluate(x_test, y_test))