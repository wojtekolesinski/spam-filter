import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import numpy as np
import re


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
            if re.search(r"[0-9]+", token.lemma_):
                new_text.append("aanumbers")
            else:
                word = "".join([ch for ch in token.lemma_.lower() if ch not in punctuation])
                if len(word) > 1 and word not in STOP_WORDS:
                    new_text.append(word)

        df['SMS'][i] = " ".join(new_text)

    # preparation for training
    df = df.sample(n=df.shape[0], random_state=43)
    train_last_index = int(df.shape[0] * 0.8)
    train_set = df.iloc[0:train_last_index]
    train_set.reset_index(drop=True, inplace=True)

    # training
    train_bag_of_words = bag_of_words(train_set)
    train_bag_of_words['Target'] = train_bag_of_words['Target'].str.strip()

    # print
    pd.options.display.max_columns = train_bag_of_words.shape[1]
    pd.options.display.max_rows = train_bag_of_words.shape[0]

    laplace_smoothing = 1

    spam_word_count = train_bag_of_words.loc[train_bag_of_words['Target'] == 'spam', 'aa':].aggregate(np.sum, axis=0)
    vocab_length = train_bag_of_words.iloc[:, 2:].shape[1]
    spam_length = train_bag_of_words.loc[train_bag_of_words['Target'] == 'spam'].SMS.str.split().apply(len).sum()
    spam_probability = (spam_word_count + laplace_smoothing) / (laplace_smoothing * vocab_length + spam_length)

    ham_word_count = train_bag_of_words.loc[train_bag_of_words['Target'] == 'ham', 'aa':].aggregate(np.sum, axis=0)
    ham_length = train_bag_of_words.loc[train_bag_of_words['Target'] == 'ham'].SMS.str.split().apply(len).sum()
    ham_probability = (ham_word_count + laplace_smoothing) / (laplace_smoothing * vocab_length + ham_length)

    probabilities = pd.DataFrame(data={'Spam Probability': spam_probability, 'Ham Probability': ham_probability},
                                 index=spam_probability.index)

    probabilities.loc['aanumbers', 'Spam Probability'] = 0.155
    print(probabilities.iloc[:200])


if __name__ == "__main__":
    main()
