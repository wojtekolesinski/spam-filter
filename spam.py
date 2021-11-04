import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


def bag_of_words(data):
    vocab = set()
    for row in data['SMS']:
        vocab.update(row.split())
    col_index = list(vocab)
    col_index.sort()

    bag = pd.DataFrame(0, index=range(len(data)), columns=col_index)
    for i, row in data['SMS'].iteritems():
        for word in row.split():
            if word in vocab:
                bag.loc[i, word] = 1

    return pd.concat([data, bag], axis=1)


def main():
    # loading data
    df = pd.read_csv("spam.csv", encoding='iso-8859-1', header=0, usecols=[0, 1], dtype=str)
    df.columns = ['Target', 'SMS']
    en_sm_model = spacy.load("en_core_web_sm")

    # preprocessing
    for i, row in df.iterrows():
        text = row['SMS'].lower()
        doc = en_sm_model(text)
        new_text = []
        for token in doc:
            if any(ch.isdigit() for ch in token.text):
                new_text.append("aanumbers")
            elif len(token.lemma_) > 1 and token.lemma_ not in STOP_WORDS \
                    and not any(ch in punctuation for ch in token.text):
                new_text.append(token.lemma_)

        df['SMS'][i] = " ".join(new_text)

    # preparation for training
    df = df.sample(n=df.shape[0], random_state=43)
    train_last_index = int(df.shape[0] * 0.8)
    train_set = df.iloc[0:train_last_index]
    train_set.reset_index(drop=True, inplace=True)

    # training
    train_bag_of_words = bag_of_words(train_set)

    # print
    pd.options.display.max_columns = train_bag_of_words.shape[1]
    pd.options.display.max_rows = train_bag_of_words.shape[0]
    print(train_bag_of_words.iloc[:200, :50])


if __name__ == "__main__":
    main()
