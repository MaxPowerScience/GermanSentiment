from perceptron import train_network
from preprocessingData import prepare_data
from tflearn.data_utils import to_categorical

def classify():
    trainX, trainY, testX, testY = prepare_data()
    train_network(trainX, trainY)

def main():
    print('test')



if __name__ == "__main__":
    main()