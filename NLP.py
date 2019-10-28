import numpy as np
import pandas as pd
import nltk
import re

data = pd.read_csv('data.csv')

# converting string to list
text = data.loc[0, 'Link_1'].strip('][')


def multi_split(text, seps):
    '''
    split string base on multiple separators
    '''
    default_sep = seps[0]

    # make all other seps equal to the default
    for sep in seps[1:]:
        text = text.replace(sep, default_sep)

    return [i.strip() for i in text.split(default_sep)]


text = multi_split(text, ('", ', "', "))

# Remove leading ' or "
for num, i in enumerate(text):
    text[num] = re.sub('^["\']', '', i)

link_1 = pd.DataFrame(text, columns=['text'])

# Counting vocabulary of words
link_1.shape  # num of paragraphs
link_1['text'].str.len()  # num characters for each paragraph

# Delete paragraphs with 0 characters
link_1 = link_1.loc[link_1['text'].str.len() > 0, :]
link_1['text'].str.len().max()  # 704
link_1['text'].str.len().min()  # 2

# number of tokens for each string
link_1['text'].str.split().str.len().max()  # 118

# how many times a digit occurs in each string
digit = link_1.loc[link_1['text'].str.count(r'\d') > 0, :]
link_1.loc[link_1['text'].str.count(r'\d') > 0, :]['text'].str.findall(r'\d')

# print string with digits
for i in range(digit.shape[0]):
    print('\n\n', digit.iloc[i, 0])

# unique tokens
set(link_1.loc[0, 'text'].split())

# Frequency of words
dist = nltk.FreqDist(' '.join(text).split())
len(dist)  # number of unique tokens in all the text

vocabl = dist.keys()

freqwords = [w for w in vocabl if len(w) > 3 and dist[w] > 20]

# Normalization and stemming (going back to root)
porter = nltk.PorterStemmer()
stemmed_text = [porter.stem(t) for t in ' '.join(text).split()]

len(nltk.FreqDist(stemmed_text))

# Lemmatization: Stemming, but resulting stems are all valid words
nltk.download('wordnet')
WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in ' '.join(text).split()]

# Tikenization
nltk.download('punkt')
nltk.word_tokenize(' '.join(text))

# create sentences
sentences = nltk.sent_tokenize(' '.join(text))
# add dot at the end of title
len(sentences)

# POS (Part-of-speech) tagging
# if you're interested in extracting specific tags (word classes / types)
nltk.download('tagsets')
nltk.help.upenn_tagset('')  # help tags meaning
