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

def main():
    read_emoticons()
    #get_word_list()

if __name__ == "__main__":
    main()