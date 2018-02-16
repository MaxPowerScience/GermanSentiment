"""Author: Manuel Reinbold, Maximilian Renk
Date: 21/11/17
Version: 1.0
"""

from xml.dom import minidom

# Delivers the unpreprocessed raw data for all social media comments.
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

    # Sort texts according to sentiment
    sorted_texts, sorted_sentiments = sort_text_by_sentiment(texts, sentiments)

    return sorted_texts, sorted_sentiments

# Reads data from a xml file and return text and sentiment of social media comment.
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

# Sorting the texts according to their sentiments (order: positive, negative, neutral).
def sort_text_by_sentiment(texts, sentiments):
    sorted_positive_texts, sorted_negative_texts, sorted_neutral_texts = [], [], []
    sorted_positive_sentiment, sorted_negative_sentiment, sorted_neutral_sentiment = [], [], []

    for idx, text in enumerate(texts):
        if sentiments[idx] == 'positive':
            sorted_positive_texts.append(text)
            sorted_positive_sentiment.append(sentiments[idx])
        elif sentiments[idx] == 'negative':
            sorted_negative_texts.append(text)
            sorted_negative_sentiment.append(sentiments[idx])
        else:
            sorted_neutral_texts.append(text)
            sorted_neutral_sentiment.append(sentiments[idx])

    sorted_texts = sorted_positive_texts + sorted_negative_texts + sorted_neutral_texts
    sorted_sentiments = sorted_positive_sentiment + sorted_negative_sentiment + sorted_neutral_sentiment

    return sorted_texts, sorted_sentiments

def main():
    print("Please execute some code")

if __name__ == "__main__":
    main()