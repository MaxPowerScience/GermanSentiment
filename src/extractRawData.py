"""Author: Manuel Reinbold, Maximilian Renk
Date: 21/11/17
Version: 1.0
"""

from xml.dom import minidom

def get_raw_data():
    # parse an xml file by name
    dev_path = '../data/germeval/dev_v1.4.xml'
    train_path = '../data/germeval/train_v1.4.xml'

    # Read text and sentiment from xml
    texts_dev, sentiments_dev = get_data_from_xml(dev_path)
    texts_train, sentiments_train = get_data_from_xml(train_path)

    # Concatenate arrays
    texts = texts_dev + texts_train
    sentiments = sentiments_dev + sentiments_train

    return texts, sentiments

def get_data_from_xml(filepath):
    my_doc = minidom.parse(filepath)
    document_items = my_doc.getElementsByTagName('Document')
    texts, sentiments = [], []
    for document in document_items:
        for child in document.childNodes:
            if child.nodeName == 'sentiment':
                sentiments.append(child.firstChild.data)
            if child.nodeName == 'text':
                texts.append(child.firstChild.data)

    return texts, sentiments

def main():
    print("Please execute some code")

if __name__ == "__main__":
    main()