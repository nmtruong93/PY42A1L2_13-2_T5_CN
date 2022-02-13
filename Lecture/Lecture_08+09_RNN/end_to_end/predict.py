import _pickle
import json

from .features_extraction import get_bow_vector

with open('model.p', 'rb') as file:
    MODEL = _pickle.load(file)

with open('../data/dictionary.json', 'r') as file:
    dictionary = json.load(file)

with open('../data/label2idx.json', 'r') as file:
    label2idx = json.load(file)

idx2label = {}
for label, idx in label2idx.items:
    idx2label[idx] = label


def predict(text):
    vector = get_bow_vector(text, dictionary)
    result = MODEL.predict(vector)

    return idx2label[result]
