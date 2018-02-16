import csv
import re
import numpy as np
import os
import zipfile

from tflearn.data_utils import to_categorical
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

def preprocess_texts(texts):
    cleaned_texts = []
    for text in texts:
        cleaned_texts.append(get_feature_string(process_text(tag_emoticons(text))))

    return cleaned_texts

def process_text(text):
    #Convert to lower case
    text = text.lower()
    #Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    #Convert @username to AT_USER
    text = re.sub('@[^\s]+','AT_USER',text)
    #Remove additional white spaces
    text = re.sub('[\s]+', ' ', text)
    #Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    #trim
    text = text.strip('\'"')
    return text

def get_stop_word_list(stop_word_list_file_name):
    #read the stopwords file and build a list
    stop_words = []
    fp = open(stop_word_list_file_name, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stop_words.append(word)
        line = fp.readline()
    fp.close()
    return stop_words

def replace_duplicate_characters(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def get_feature_string(text):
    feature_string = ""
    #split text into words
    words = text.split()
    for w in words:
        #replace two or more with two occurrences
        w = replace_duplicate_characters(w)
        #strip punctuation
        w = w.strip('\'"?!,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in get_stop_word_list('stopwords.txt') or val is None):
            continue
        else:
            feature_string += w.lower() + " "

    return feature_string.strip()

def prepare_data():
    all_texts, pos_texts, neu_texts, neg_texts, sentiments = get_raw_data()

    return separate_test_and_training_data(pos_texts, neu_texts, neg_texts, all_texts)

def create_ids(cleaned_texts, ids_name, max_seq_length):
    words_list = read_word_list()
    unknown_word = 1924893

    ids = np.zeros((len(cleaned_texts), max_seq_length), dtype='int32')
    file_counter = 0
    for text in cleaned_texts:
        index_counter = 0
        split = text.split()
        for word in split:
            try:
                ids[file_counter][index_counter] = words_list.index(word)
            except ValueError:
                ids[file_counter][index_counter] = unknown_word  # Vector for unkown words
            index_counter = index_counter + 1
            if index_counter >= max_seq_length:
                break
        file_counter = file_counter + 1
        print('Texts %s', file_counter)

    np.save(ids_name, ids)

    return ids


def read_word_list():
    word_list = []

    with open('../resources/wordList.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            word_list.append(line.strip('\n'))

    return word_list

# Load or create ids matrix
def get_ids_matrix(texts):
    ids_name = 'idsMatrix'

    path_id_matrix = '../resources/' + ids_name + '.npy'
    if os.path.isfile(path_id_matrix):
        ids = np.load(path_id_matrix)
    else:
        max_seq_length = 40
        cleaned_texts = preprocess_texts(texts)
        ids = create_ids(cleaned_texts, ids_name, max_seq_length)

    return ids

def separate_test_and_training_data(pos_texts, neu_texts, neg_texts, all_texts):

    # Split data in train and test
    percentage_train_data = 0.8
    percentage_test_data = 1 - percentage_train_data

    number_of_positive_train = round(len(pos_texts) * percentage_train_data)
    number_of_negative_train = round(len(neg_texts) * percentage_train_data)
    number_of_neutral_train = round(len(neu_texts) * percentage_train_data)

    number_of_positive_test = len(pos_texts) - number_of_positive_train
    number_of_negative_test = len(neg_texts) - number_of_negative_train
    number_of_neutral_test = len(neu_texts) - number_of_neutral_train

    lower_bound_pos_train = 0
    upper_bound_pos_train = lower_bound_pos_train + number_of_positive_train
    lower_bound_pos_test = upper_bound_pos_train + 1
    upper_bound_pos_test = lower_bound_pos_test + number_of_positive_test

    lower_bound_neg_train = 0
    upper_bound_neg_train = lower_bound_neg_train + number_of_negative_train
    lower_bound_neg_test = upper_bound_neg_train + 1
    upper_bound_neg_test = lower_bound_neg_test + number_of_negative_test

    lower_bound_neu_train = 0
    upper_bound_neu_train = lower_bound_neu_train + number_of_neutral_train
    lower_bound_neu_test = upper_bound_neu_train + 1
    upper_bound_neu_test = lower_bound_neu_test + number_of_neutral_test

    ids = get_ids_matrix(all_texts)

    positive_train = ids[lower_bound_pos_train:upper_bound_pos_train]
    positive_test = ids[lower_bound_pos_test:upper_bound_pos_test]
    negative_train = ids[lower_bound_neg_train:upper_bound_neg_train]
    negative_test = ids[lower_bound_neg_test:upper_bound_neg_test]
    neutral_train = ids[lower_bound_neu_train:upper_bound_neu_train]
    neutral_test = ids[lower_bound_neu_test:upper_bound_neu_test]

    pos_labels_train = [1,0,0] * len(positive_train)
    pos_labels_test = [1,0,0] * len(positive_test)
    neg_labels_train = [0,0,1] * len(negative_train)
    neg_labels_test = [0,0,1] * len(negative_test)
    neu_labels_train = [0,1,0] * len(neutral_train)
    neu_labels_test = [0,1,0] * len(neutral_test)

    trainX = np.concatenate([positive_train, negative_train, neutral_train])
    #trainY = to_categorical(pos_labels_train + neg_labels_train + neu_labels_train, nb_classes=3)
    trainY = np.concatenate([pos_labels_train, neg_labels_train, neu_labels_train])

    testX = np.concatenate([positive_test, negative_test, neutral_test])
    #testY = to_categorical(pos_labels_test + neg_labels_test + neu_labels_test, nb_classes=3)
    testY = np.concatenate([pos_labels_test, neg_labels_test, neu_labels_test])

    return trainX, trainY, testX, testY

def main():
    prepare_data()
    #get_word_list()

if __name__ == "__main__":
    main()