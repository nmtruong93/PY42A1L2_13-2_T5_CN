import re
from collections import defaultdict
import json
import os
from glob import glob

import bs4
import demoji
from pyvi.ViTokenizer import tokenize

demoji.download_codes()


def del_html(text):
    soup = bs4.BeautifulSoup(text)
    return soup.get_text(' ')


def del_link(text):
    link = r'http[\S]*'
    text = re.sub(link, ' ', str(text))
    return text


def del_punctuation(doc):
    pattern = r'[\,\.\/\\\!\@\#\+\"\'\;\)\(\“\”\\\-\:…&><=\-\%\|\^\$\&\)\(\[\]\{\}\?\*\•]'
    record = re.sub(pattern, ' ', doc)
    return re.sub(r'\n', ' ', record)


def del_emoji(text):
    return demoji.replace(text, '')


def del_numbers(text):
    return re.sub(r'\d+', ' ', text)


def del_space(doc):
    space_pattern = r'\s+'
    return re.sub(space_pattern, ' ', doc.lower())


def text_token(text):
    return tokenize(text)


def clean_text(text):
    text = del_html(text)
    text = del_link(text)
    text = del_numbers(text)
    text = del_emoji(text)
    text = del_punctuation(text)
    text = del_space(text)
    return text_token(text)


def get_cleaned_data(link, path_output, select_topics):
    data = defaultdict(list)
    topics = os.listdir(link)
    for topic in topics:
        path = os.path.join(link, topic)
        if topic in select_topics:
            paths_text = glob(path + '/*')
            for p in paths_text:
                with open(p, 'r', encoding='utf-16') as file:
                    txt = file.read()
                data[topic].append(clean_text(txt))

    with open(path_output, 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    # paths_train = '../data/data_text_classification/new train'
    # paths_test = '../data/data_text_classification/new test'
    #
    # t = ['Am nhac', 'Am thuc', 'Bat dong san', 'Bong da', 'Chung khoan']
    # get_cleaned_data(paths_train, '../data/cleaned_train_data.json', t)
    # print('Train: Done')
    # get_cleaned_data(paths_test, '../data/cleaned_test_data.json', t)

    link_train_data = '../data/cleaned_train_data.json'
    link_test_data = '../data/cleaned_test_data.json'

    with open(link_train_data, 'r') as file:
        data_train = json.load(file)

    # with open(link_test_data, 'r') as file:
    #     data_test = json.load(file)

    vocabulary = []
    for topic in data_train:
        print(topic)
        for record in data_train[topic]:
            for word in record.split():
                if word not in vocabulary:
                    vocabulary.append(word)

    dictionary = {}
    for ind, word in enumerate(vocabulary):
        dictionary[word] = ind

    with open('../data/dictionary.json', 'w') as file:
        json.dump(dictionary, file)
