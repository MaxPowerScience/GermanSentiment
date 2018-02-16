from extractRawData import get_raw_data

def get_word_list():
    texts, sentiments = get_raw_data()


    #Read Emoticions from file

    #Emoticons handling







    for text in texts:
        words = text.split()
        for word in words:
            print(word.strip('\'"?!,.:'))

def readEmoticons():
    emoticonsPath = '../resources/germeval/train_v1.4.xml'

def main():
    get_word_list()

if __name__ == "__main__":
    main()