import csv
import re
import numpy as np
import os
import zipfile

from extractRawData import get_raw_data

# Reads all information about latin only emoticons
def read_emoticons():
    emoticons_path = '../resources/emoticonsWithSentiment.csv'

    emoticons = []
    emoticons_tags = []
    sentiments = []

    with open(emoticons_path, 'rt', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            emoticons.append(row[0])
            if len(row) == 3:
                emoticons_tags.append(row[1])
                sentiments.append(row[2])
            else:
                sentiments.append([row[2], row[3]])

    return  emoticons, emoticons_tags, sentiments


def tag_emoticons(text, emoticons_dict):

    split = text.split()
    for word in split:
        if word in emoticons_dict:
            text = text.replace(word, emoticons_dict.get(word))

    return text

def preprocess_text():
    all_texts, pos_texts, neu_texts, neg_texts, sentiments = get_raw_data()
    words_list = read_word_list()
    number_of_texts = 21801
    max_seq_length = 40
    ids_name = 'idsMatrix'
    ids = create_ids(number_of_texts, max_seq_length, words_list, pos_texts, neu_texts, neg_texts, ids_name)
    separate_test_and_training_data(pos_texts, neu_texts, neg_texts, words_list )
    # Read Emoticions from file
    emoticons, emoticons_tags, _ = read_emoticons()

    print(emoticons)
    print(emoticons_tags)

    # Emoticons handling

    # Schleife über alle Daten
    # Einen Datensatz
    # Schleife über datensatz
    # Entfernst überflüssig und ersetzt durch tags
    emoticons_dict = dict(zip(emoticons, emoticons_tags))
    for text in all_texts:
        print(tag_emoticons(text, emoticons_dict))

    return text

def create_ids(num_files, max_seq_length, words_list, pos_texts, neu_texts, neg_texts, ids_name):
    words_list = np.load('../wordlist/wordsList.npy')
    words_list = words_list.tolist()

    unknown_word = 1924893

    ids = np.zeros((num_files, max_seq_length), dtype='int32')
    file_counter = 0
    for pos_text in pos_texts:
        index_counter = 0
        cleaned_text = preprocess_text(pos_text)
        split = cleaned_text.split()
        for word in split:
            try:
                ids[file_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[file_counter][index_counter] = unknown_word  # Vector for unkown words
            index_counter = index_counter + 1
            if index_counter >= max_seq_length:
                break
        file_counter = file_counter + 1
        print("Positive ")
        print(file_counter)

    for neg_text in neg_texts:
        index_counter = 0
        cleaned_text = preprocess_text(neg_text)
        split = cleaned_text.split()
        for word in split:
            try:
                ids[file_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[file_counter][index_counter] = unknown_word  # Vector for unkown words
            index_counter = index_counter + 1
            if index_counter >= max_seq_length:
                break
        file_counter = file_counter + 1
        print("Negative ")
        print(file_counter)

    for neu_text in neu_texts:
        index_counter = 0
        cleaned_text = preprocess_text(neu_text)
        split = cleaned_text.split()
        for word in split:
            try:
                ids[file_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[file_counter][index_counter] = unknown_word  # Vector for unknown words
            if index_counter >= max_seq_length:
                break
        file_counter = file_counter + 1
        print("Neutral ")
        print(file_counter)

    np.save(ids_name, ids)

    return ids

def separate_test_and_training_data(pos_texts, neu_texts, neg_texts, word_list, ids_name):

    # Load or create ids matrix
    load_ids = True
    number_of_texts = 21801
    maxSeqLength = 40
    if load_ids:
        path_id_matrix = '../idMatrix/' + ids_name + '.npy'
        if not os.path.exists(path_id_matrix):
            with zipfile.ZipFile('../idMatrix/' + ids_name + '.zip', 'r') as zip_ref:
                zip_ref.extractall("../idMatrix/")
        ids_train = np.load(path_id_matrix)
    else:
        ids_train = create_ids(number_of_texts, maxSeqLength, word_list, pos_texts, neg_texts, neu_texts,
                               ids_name)

    upper_pos_train = 1326
    upper_neg_train = 6960
    upper_neu_train = 21800
    upper_pos_test = 0
    upper_neg_test = 0
    upper_neu_test = 0

    #Split data in train and test
    percentage_train_data = 0.8
    percentage_test_data = 1 - percentage_train_data

    # 400000 100000
    number_of_tweets_train = round(percentage_train_data * number_of_tweets_class)
    number_of_tweets_test = round(percentage_test_data * number_of_tweets_class)

    # 0 - 1326
    pos_texts_train = ids_train[0:upper_pos_train]
    pos_labels_train = [0] * len(pos_texts_train)

    # -
    pos_texts_test = ids_train[upper_neu_train+1:upper_pos_test]
    pos_labels_test = [0] * len(pos_texts_test)

    # 1327 - 6960
    neg_texts_train = ids_train[upper_pos_train+1:upper_neg_train]
    neg_labels_train = [1] * len(neg_texts_train)

    # -
    neg_texts_test = ids_train[upper_pos_test+1:upper_neg_test]
    neg_labels_test = [1] * len(neg_texts_test)

    # 6961 - 21800
    neu_texts_train = ids_train[upper_neg_train+1:upper_neu_train]
    neu_labels_train = [2] * len(neu_texts_train)

    # -
    neu_texts_test = ids_train[upper_neg_test+1:upper_neu_test]
    neu_labels_test = [2] * len(neu_texts_test)

    trainX = np.concatenate([pos_tweets_train, neg_tweets_train])
    trainY = to_categorical(pos_labels_train + neg_labels_train, nb_classes=2)

    testX = np.concatenate([pos_tweets_test, neg_tweets_test])
    testY = to_categorical(pos_labels_test + neg_labels_test, nb_classes=2)

    return trainX, trainY, testX, testY

def main():
    preprocess_text()
    #get_word_list()

if __name__ == "__main__":
    main()