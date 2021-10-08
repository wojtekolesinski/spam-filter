import pandas as pd
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en.punctuation import TOKENIZER_INFIXES
from spacy.glossary import GLOSSARY

nlp = spacy.load("en_core_web_sm")
data = pd.read_csv('spam.csv', encoding='iso-8859-1')

# Will be using just the first two columns
data = data.iloc[:, :2]
data.columns = ['Target', 'SMS']

# data['SMS'] = data['SMS'].str.lower()
i = 0
print(data.size)
print(data.head())
for index, row in data.iterrows():
    row['SMS'] = row['SMS'].lower()
    lemmas = (token.lemma_ for token in nlp(row['SMS'])) # lemmatization
    no_punctuation = (word for word in lemmas if word not in TOKENIZER_INFIXES)
    no_pos_tags = (word for word in no_punctuation if word not in GLOSSARY.keys())
    aanumbers = []
    for word in no_pos_tags:
        if re.match('.*\\d.*', word) is not None:
            aanumbers.append(word)
        else:
            aanumbers.append('aanumbers')
    no_single_letters = (word for word in aanumbers if word.len() != 1)
    row['SMS'] = ' '.join(no_single_letters)

    i += 1
    if i == 205:
        break


print(data.head())
