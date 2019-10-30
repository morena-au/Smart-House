import re

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn

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
# TODO: code numbers as <_digit_> tokens
# TODO: code date as <_date_> tokens
digit = link_1.loc[link_1['text'].str.count(r'\d') > 0, :]
link_1.loc[link_1['text'].str.count(r'\d') > 0, :]['text'].str.findall(r'\d')

# print string with digits
for i in range(digit.shape[0]):
    print('\n\n', digit.iloc[i, 0])

# unique tokens
set(link_1.loc[0, 'text'].split())

# Frequency of words in all the text
dist = nltk.FreqDist(' '.join(text).split())
len(dist)  # number of unique tokens in all the text >> 1384

# Sort frequency
sorted(dist.items(), key=lambda x: x[1], reverse=True)

vocabl = dist.keys()
freqwords = [w for w in vocabl if len(w) > 3 and dist[w] > 20]
# TODO: code product Siri, Alexa, Google Assistant names as <product_name>
# TODO: code companies Apple, Google, Amazon, names as <company_name>

# remove stopwords
# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
             "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
             "during", "each", "few", "for", "from", "further", "had", "has", "have",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
             "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
             "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me",
             "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
             "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
             "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
             "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
             "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
             "we're", "we've", "were", "what", "what's", "when", "when's", "where",
             "where's", "which", "while", "who", "who's", "whom", "why", "why's",
             "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
             "yours", "yourself", "yourselves", "however"]

for num, paragraph in enumerate(text):
    text[num] = ' '.join(
        [w for w in paragraph.strip().split() if not w in stopwords])

# Normalization and stemming (going back to root)
porter = nltk.PorterStemmer()
stemmed_text = [porter.stem(t) for t in ' '.join(text).split()]

# number of unique stemmed tokens in all the text >> 1189
stemmed_dist = nltk.FreqDist(stemmed_text)
len(stemmed_dist)
[w for w in stemmed_dist.keys() if len(w) > 3 and stemmed_dist[w] > 10]
sorted(stemmed_dist.items(), key=lambda x: x[1], reverse=True)

# Lemmatization: Stemming, but resulting stems are all valid words
# nltk.download('wordnet')
WNlemma = nltk.WordNetLemmatizer()
lemma_text = [WNlemma.lemmatize(t) for t in ' '.join(text).split()]

# number of unique lemmatize tokens in all the text >> 1335
len(nltk.FreqDist(lemma_text))

# Tokenization
# nltk.download('punkt')
# Tokenize a string to split off punctuation other than periods
nltk.word_tokenize(' '.join(text))

# create sentences: change grain of the analysis
sentences = nltk.sent_tokenize(' '.join(text))
# TODO: add dot at the end of title (headline)
len(sentences)

# TF - IDF
# min document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(text)
feature_names = np.array(vect.get_feature_names())

text_vectorized = vect.transform(text)

sorted_tfidf_index = text_vectorized.max(0).toarray()[0].argsort()

# Smallest: words commonly used across all documents and rarely used in the particular document.
print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
# Largest: Term that appears frequently in a particular document, but not often in the corpus.
# "paragraph hedline - topics"
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

# TOPIC MODELLING


# INFORAMTION EXTRACTION
# N-GRAM add context by ading sequences of word.

# POS (Part-of-speech) tagging >> Information Extraction
# if you're interested in extracting specific tags (word classes / types)
# nltk.download('tagsets')
nltk.help.upenn_tagset('')  # help tags meaning
# Document Similarity
# nltk.download('averaged_perceptron_tagger')


def convert_tag(tag):
    '''
    Convert tag given by nltk.pos_tag to the wordnet.synsets'tag
    '''

    # n = noun, v = verb, a = adjective, r = adverb
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}

    # not all tags are being converted
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

    tag_dict['N']


def doc_to_synsets(doc):
    '''
    Retuns a list of synsets in document
    '''

    token = nltk.word_tokenize(doc)
    word_tag = nltk.pos_tag(token)

    synsets = []

    for word, tag in word_tag:
        tag = convert_tag(tag)
        # Synset simple interface in nltk to look up words in WordNet
        # synset instance are grupings of synonymous words
        synset = wn.synsets(word, pos=tag)
        if len(synset) != 0:
            synsets.append(synset[0])
        else:
            continue

    return synsets


def similarity_score(s1, s2):
    '''
    Calculate the normalized similarity score.
    Input: s1, s2, list of synsets from doc_to_synsets
    Output: normalized similarity score of s1 onto s2
    '''

    largest_sim_values = []
    # similarity score
    for syn1 in s1:
        similarity_values = []
        for syn2 in s2:
            sim_val = wn.path_similarity(syn1, syn2)
            if sim_val is not None:
                similarity_values.append(sim_val)
        if len(similarity_values) != 0:
            largest_sim_values.append(max(similarity_values))

    return sum(largest_sim_values) / len(largest_sim_values)


s1 = doc_to_synsets(text[0])
s2 = doc_to_synsets(text[1])

similarity_score(s1, s2)
