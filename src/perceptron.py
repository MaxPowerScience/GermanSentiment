import os
import tflearn
import datetime

def create_perceptron():
    net = tflearn.input_data([None, 40])
    net = tflearn.embedding(net, input_dim=400000, output_dim=128)
    net = tflearn.fully_connected(net, 100, activation='relu')
    net = tflearn.fully_connected(net, 20, activation='relu')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1,
                             loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='../tensorboard/tensorboard_fully')

def train_network(trainX, trainY):
    batch_size = 64

    model = create_perceptron()
    save_folder = '../models/fully/'
    save_name = 'fully_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.tfl'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = save_folder + save_name

    n_epoch = 3
    for i in range(0,n_epoch):
        trainX, trainY = tflearn.data_utils.shuffle(trainX, trainY)
        model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=batch_size, n_epoch=1)

    model.save(save_path)
    return model

def main():
    create_perceptron()

if __name__ == "__main__":
    main()