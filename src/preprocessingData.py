import csv
import re
import numpy as np
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
    texts, sentiments = get_raw_data()

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
    for text in texts:
        print(tag_emoticons(text, emoticons_dict))


def read_word_list():
    word_list = []

    with open('../resources/wordList.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            word_list.append(line.strip('\n'))

    return word_list

def create_ids(num_files, max_seq_length, words_list, pos_texts, neg_texts, neu_texts, ids_name):
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

def main():
    #preprocess_text()
    #get_word_list()
    read_word_list()

if __name__ == "__main__":
    main()