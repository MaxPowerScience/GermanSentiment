import datetime
import tflearn
import os
import numpy as np

def train_network(trainX, trainY, model):
    batch_size = 32

    save_folder = '../models/'
    save_name = 'model_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.tfl'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = save_folder + save_name

    n_epoch = 500
    for i in range(0,n_epoch):
        trainX, trainY = tflearn.data_utils.shuffle(trainX, trainY)
        model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=batch_size, n_epoch=1)

    model.save(save_path)
    return model

def test_network(testX, testY, model):

    # Transform acutal class labels (1 for positiv and 0 for negative)
    actualClasses = []
    for test in testY:
        actualClasses.append(1) if test[0] == 1 else actualClasses.append(0)

    # Predict class for test data

    pred = np.array(model.predict(testX))
    predictions = (np.array(model.predict(testX))[:,1] >= 0.5).astype(np.int_)

    # Print classification report and confusion matrix
    #print(sklearn.metrics.classification_report(actualClasses, predictions))
    #print(sklearn.metrics.confusion_matrix(actualClasses, predictions))

    # Accuracy of test prediction
    test_accuracy = np.mean(predictions == testY[:,1], axis=0)
    print("Test accuracy: ", test_accuracy)

def main():
    print()

if __name__ == "__main__":
    main()