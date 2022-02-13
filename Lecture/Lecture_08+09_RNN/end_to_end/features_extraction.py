import json
import numpy as np
import _pickle


def get_bow_vector(text, dictionary):
    vector = np.zeros(len(dictionary))
    for word in text.split(' '):
        if word in dictionary:
            vector[dictionary[word]] += 1

    return vector


if __name__ == '__main__':
    with open('../data/dictionary.json', 'r') as file:
        dictionary = json.load(file)

    vector_train = []
    label_train = []
    vector_test = []
    label_test = []

    link_train_data = '../data/cleaned_train_data.json'
    link_test_data = '../data/cleaned_test_data.json'

    with open(link_train_data, 'r') as file:
        data_train = json.load(file)

    # with open(link_test_data, 'r') as file:
    #     data_test = json.load(file)

    label_dict = {}
    for idx, topic in enumerate(list(data_train.keys())):
        label_dict[topic] = idx

    with open('../data/label2idx.json', 'w') as file:
        json.dump(label_dict, file)

    # for topic in data_train:
    #     for record in data_train[topic]:
    #         vector_train.append(get_bow_vector(record, dictionary))
    #         label_train.append(label_dict[topic])
    #
    # for topic in data_test:
    #     for record in data_test[topic]:
    #         vector_test.append(get_bow_vector(record, dictionary))
    #         label_test.append(label_dict[topic])
    #
    # with open('../data/bow_train.p', 'wb') as file:
    #     _pickle.dump({'vector': vector_train, 'label': label_train}, file)
    #
    # with open('../data/bow_test.p', 'wb') as file:
    #     _pickle.dump({'vector': vector_test, 'label': label_test}, file)
