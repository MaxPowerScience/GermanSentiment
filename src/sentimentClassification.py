from perceptron import train_network, create_perceptron, test_network
from preprocessingData import get_ids_matrix, separate_test_and_training_data, read_word_list, stem_word_list
from extractRawData import get_raw_data


def main():
    all_texts, pos_texts, neu_texts, neg_texts, sentiments = get_raw_data()
    dictionary = read_word_list()
    #dictionary = stem_word_list(dictionary)
    ids = get_ids_matrix(all_texts, dictionary)
    max_seq_length = len(ids[0])
    trainX, trainY, testX, testY = separate_test_and_training_data(pos_texts, neg_texts, neu_texts, ids)
    model = create_perceptron(max_seq_length, len(dictionary))
    train_network(trainX, trainY, model)

    #snapshot_name = "perceptron_20180220-152036.tfl"
    #load_folder = '../models/perceptron/'
    #load_path = load_folder + snapshot_name
    #model.load(load_path)
    #print('Model loaded')

    #test_network(testX, testY, model)


if __name__ == "__main__":

    main()