from perceptron import train_network, create_perceptron
from preprocessingData import get_ids_matrix, separate_test_and_training_data, read_word_list
from extractRawData import get_raw_data
from tflearn.data_utils import to_categorical

def main():
    all_texts, pos_texts, neu_texts, neg_texts, sentiments = get_raw_data()
    dictionary = read_word_list()
    ids = get_ids_matrix(all_texts, dictionary)
    max_seq_length = len(ids[0])
    trainX, trainY, testX, testY = separate_test_and_training_data(pos_texts, neu_texts, neg_texts, ids)
    model = create_perceptron(max_seq_length, len(dictionary))
    train_network(trainX, trainY, model)

if __name__ == "__main__":
    main()