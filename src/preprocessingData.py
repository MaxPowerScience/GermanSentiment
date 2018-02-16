import csv
from extractRawData import get_raw_data

def get_word_list():
    texts, sentiments = get_raw_data()

    #Read Emoticions from file
    emoticons, emoticonSentiment = read_emoticons()

    #Emoticons handling

    for text in texts:
        words = text.split()
        for word in words:
            print(word.strip('\'"?!,.:'))

def read_emoticons():
    emoticonsPath = '../resources/emoticonsWithSentiment.csv'
    emoticons = []
    sentiments = []

    with open(emoticonsPath, 'rt', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            numberOfRowValue = len(row)
            emoticons.append(row[0])
            if numberOfRowValue == 2:
                sentiments.append(row[1])
            else:
                sentiments.append([row[1], row[2]])

    return  emoticons, sentiments

def main():
    read_emoticons()
    #get_word_list()

if __name__ == "__main__":
    main()