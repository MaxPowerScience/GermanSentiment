import csv
import re
import numpy as np
import os
import nltk.stem

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

def preprocess_texts(texts):
    cleaned_texts = []
    emoticons, emoticons_tags, _ = read_emoticons()
    emoticons_dict = dict(zip(emoticons, emoticons_tags))
    stop_words = get_stop_word_list('../resources/stopwords.txt')
    stemmer = nltk.stem.SnowballStemmer('german')
    for text in texts:
        cleaned_texts.append(process_text(text, emoticons_dict, stop_words, stemmer))

    return cleaned_texts

def process_text(text, emoticons_dict, stop_words, stemmer):
    processed_text = ''
    # Convert list to set for faster lookup
    stop_words = set(stop_words)

    # Convert to lower case
    text = text.lower()
    # Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    # Convert @username to AT_USER
    text = re.sub('@[^\s]+', 'AT_USER', text)
    # Remove additional white spaces
    text = re.sub('[\s]+', ' ', text)
    # Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # Trim
    text = text.strip('\'"')

    split_text = text.split()
    for word in split_text:
        if word in emoticons_dict:
            word = word.replace(word, emoticons_dict.get(word))
        else:
            # Replace two or more with two occurrences
            word = replace_duplicate_characters(word)
            # Strip punctuation
            word = word.strip('\'"?!,.')
            # Check if the word stats with an alphabet
            val = re.search(r"^[a-zA-ZäüöÄÜÖß][a-zA-ZäüöÄÜÖß0-9]*$", word)
            #if (word in stop_words or val is None):
            if (val is None):
                continue
        #word = stemmer.stem(word)
        processed_text += word + ' '

    return processed_text

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

def create_ids(cleaned_texts, ids_name, max_seq_length, dictionary):
    unknown_word = len(dictionary)
    dictionaryset = set(dictionary)

    ids = np.zeros((len(cleaned_texts), max_seq_length), dtype='int32')
    file_counter = 0
    for text in cleaned_texts:
        index_counter = 0
        split = text.split()
        for word in split:
            if (word in dictionaryset):
                ids[file_counter][index_counter] = dictionary.index(word)
            else:
                ids[file_counter][index_counter] = unknown_word
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

def stem_word_list(word_list):
    stemmer = nltk.stem.SnowballStemmer('german')
    stemmed_list = []
    i = 0
    with open('../resources/wordListStemmed.txt', 'w', encoding='UTF-8') as file:
        for word in word_list:
            print(i)
            stemmed_word = stemmer.stem(word)
            if not stemmed_word in stemmed_list:
                stemmed_list.append(stemmed_word)
                file.write("%s\n" % stemmed_word)
            i = i + 1

    print('length stemmed word list %d', len(stemmed_list))
    return stemmed_list


# Load or create ids matrix
def get_ids_matrix(texts, dictionary):
    ids_name = 'idsMatrixStem'
    path_id_matrix = '../resources/' + ids_name + '.npy'
    max_seq_length = 0
    if os.path.isfile(path_id_matrix):
        ids = np.load(path_id_matrix)
    else:
        cleaned_texts = preprocess_texts(texts)
        max_seq_length = int(get_max_sequence_length(cleaned_texts))
        ids = create_ids(cleaned_texts, path_id_matrix, max_seq_length, dictionary)
    return ids

def get_max_sequence_length(cleaned_texts):
    length_of_texts = []

    for text in cleaned_texts:
        text_split = text.split()
        text_split_length = len(text_split)
        length_of_texts.append(text_split_length)

    max_sequence_length = round(np.percentile(length_of_texts, 99))

    return max_sequence_length

def separate_test_and_training_data(pos_texts, neu_texts, neg_texts, ids):

    # Split data in train and test
    percentage_train_data = 1

    #number_of_positive_train = round(len(pos_texts) * percentage_train_data)
    number_of_negative_train = round(len(neg_texts) * percentage_train_data)
    number_of_neutral_train = round(len(neu_texts) * percentage_train_data)

    #number_of_positive_test = len(pos_texts) - number_of_positive_train
    number_of_negative_test = len(neg_texts) - number_of_negative_train
    number_of_neutral_test = len(neu_texts) - number_of_neutral_train

    #lower_bound_pos_train = 0
    #upper_bound_pos_train = lower_bound_pos_train + number_of_positive_train
    #lower_bound_pos_test = upper_bound_pos_train + 1
    #upper_bound_pos_test = lower_bound_pos_test + number_of_positive_test

    lower_bound_neg_train = 0
    upper_bound_neg_train = lower_bound_neg_train + number_of_negative_train
    lower_bound_neg_test = upper_bound_neg_train + 1
    upper_bound_neg_test = lower_bound_neg_test + number_of_negative_test

    lower_bound_neu_train = 0
    #upper_bound_neu_train = lower_bound_neu_train + number_of_neutral_train
    upper_bound_neu_train = lower_bound_neu_train + number_of_negative_train
    lower_bound_neu_test = upper_bound_neu_train + 1
    upper_bound_neu_test = lower_bound_neu_test + number_of_neutral_test

    #positive_train = ids[lower_bound_pos_train:upper_bound_pos_train]
    #positive_test = ids[lower_bound_pos_test:upper_bound_pos_test]
    negative_train = ids[lower_bound_neg_train:upper_bound_neg_train]
    negative_test = ids[lower_bound_neg_test:upper_bound_neg_test]
    neutral_train = ids[lower_bound_neu_train:upper_bound_neu_train]
    neutral_test = ids[lower_bound_neu_test:upper_bound_neu_test]

    #pos_labels_train = [0] * len(positive_train)
    #pos_labels_test = [0] * len(positive_test)
    neu_labels_train = [0] * len(neutral_train)
    neu_labels_test = [0] * len(neutral_test)
    neg_labels_train = [1] * len(negative_train)
    neg_labels_test = [1] * len(negative_test)

    #trainX = np.concatenate([positive_train, neutral_train, negative_train])
    trainX = np.concatenate([neutral_train, negative_train])
    #trainY = to_categorical(np.concatenate([pos_labels_train, neu_labels_train, neg_labels_train]), nb_classes=3)
    trainY = to_categorical(np.concatenate([neu_labels_train, neg_labels_train]), nb_classes=2)

    #testX = np.concatenate([positive_test, neutral_test, negative_test])
    testX = np.concatenate([neutral_test, negative_test])
    #testY = to_categorical(np.concatenate([pos_labels_test, neu_labels_test, neg_labels_test]), nb_classes=3)
    testY = to_categorical(np.concatenate([neu_labels_test, neg_labels_test]), nb_classes=2)

    return trainX, trainY, testX, testY

def main():
    all_texts, pos_texts, neu_texts, neg_texts, sentiments = get_raw_data()
    #emoticons, emoticons_tags, _ = read_emoticons()
    #emoticons_dict = dict(zip(emoticons, emoticons_tags))
    #stop_words = get_stop_word_list('../resources/stopwords.txt')
    #stemmer = nltk.stem.SnowballStemmer('german')
    #text = 'Fußballspieler'
    #print(process_text(text, emoticons_dict, stop_words, stemmer))
    #dictionary = read_word_list()
    #ids = get_ids_matrix(all_texts, dictionary)
    #trainX, trainY, testX, testY = separate_test_and_training_data(pos_texts, neu_texts, neg_texts, ids)

    #get_ids_matrix(all_texts)
    #prepare_data()
    #get_word_list()

if __name__ == "__main__":
    main()
