from extractRawData import get_raw_data

def get_word_list():
    texts, _ = get_preprocessed_data()
    for text in texts:
        words = text.split()
        for word in words:
            print(word.strip('\'"?!,.:'))

def get_preprocessed_data():
    texts, sentiment = get_raw_data()
    return texts, sentiment

def main():
    get_word_list()

if __name__ == "__main__":
    main()