import warnings
import json
import re

import gensim
import gensim.corpora as corpora
import nltk
import numpy as np
import pandas as pd
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import CoherenceModel, TfidfModel
from gensim.utils import simple_preprocess
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict

# Enable logging for gensim - optional
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


warnings.filterwarnings("ignore", category=DeprecationWarning)

# import training data

with open('train_data.json', 'r') as f:
    train_data = json.load(f)


link = []
category = []
body_par = []
comment_par = []

for item in train_data:
    link.append(item['link'])
    category.append(item['category'])
    body_par.append(item['body_par'])
    comment_par.append(item['comment_par'])


# nltk.download('stopwords')  # (run python console)
# python3 -m spacy download en  # (run in terminal)

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend([])

# Tokenize words and remove punctuations and unnecessary characters


def sent_to_words(webpages):
    pages = []
    for webpage in webpages:
        pars = []
        for paragraph in webpage:
            # simeple preprocess remove also digits
            # deacc=True removes punctuations
            pars = pars + simple_preprocess(str(paragraph), deacc=True)

        pages.append(pars)

    return pages


# turn a generator into a list
train_words = list(sent_to_words(body_par))

# num characters for each paragraph
print('Lenght words for each webpage: \n\n',
      [len(word) for word in train_words])
print('-' * 20)
print('\n')

# Train Bigram and Trigram Models
# higher threshold fewer phrases.
bigram = gensim.models.Phrases(train_words, min_count=5, threshold=40)
trigram = gensim.models.Phrases(bigram[train_words], threshold=40)

# Faster way to get a paragraph clubbed as trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# vis words with underscore
words_list = trigram_mod[bigram_mod[train_words]]

# Frequency of n-gram words
dist = nltk.FreqDist(
    [word for par in words_list for word in par if '_' in word])

# Sort frequency
print('Sorted trigrams: \n')
print(sorted(dist.items(), key=lambda x: x[1], reverse=True))
print('-'*20)
len(dist)

# Remove stopwords


def remove_stopwords(texts):
    '''
    Input: words' paragraphs
    OUtput: words' paragraphs without stop words
    '''
    par_words = list(sent_to_words(texts))

    return [[word for word in page if word not in stop_words] for page in par_words]


def make_bigrams(texts):
    '''
    Input: words' paragraphs without stop words
    Output: bigram model
    '''
    return [bigram_mod[page] for page in texts]


def make_trigrams(texts):
    '''
    Input: words' paragraphs without stop words
    Output: trigram model
    '''
    return [trigram_mod[bigram_mod[page]] for page in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']):
    '''
    Input: words' paragraphs without stop words
    OUtput: words' paragraphs without stop words and lemmatization
    https://spacy.io/api/annotation
    '''

    texts_out = []
    for page_words in texts:
        # join single words to a unique string of text
        page_text = nlp(' '.join(page_words))
        # return token (word) lemma if part of speech (pos) within allowed
        texts_out.append(
            [token.lemma_ for token in page_text if token.pos_ in allowed_postags])

    return texts_out


# Remove Stop Words
train_words_nostop = remove_stopwords(body_par)

# Form trigrams
train_words_bigrams = make_bigrams(train_words_nostop)
train_words_trigrams = make_trigrams(train_words_nostop)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv as per default
train_words_lemmatized = lemmatization(train_words_trigrams)

# Create Dictionary
id2word = corpora.Dictionary(train_words_lemmatized)

# Create Corpus
texts = train_words_lemmatized

# Term Document Frequency >> (id, freq) for each page
corpus = [id2word.doc2bow(text) for text in texts]

print('\nPrint words and frequencies in the first website:\n')
print([[(id2word[id], freq) for id, freq in page] for page in corpus[:1]])
print('-' * 20)

# Create the TF-IDF model
# Term frequency = 'n' (Occurence frequency of term in document)
# Document frequency = 't' (non-zero inverse collection frequency)
# Document lenght normalization = 'c' (cosine normalization)
tfidf = TfidfModel(corpus, smartirs='ntc')

tfidf_list = []

for page in tfidf[corpus]:
    tfidf_list = tfidf_list + \
        [(id2word[id], np.around(freq, decimals=3)) for id, freq in page]

# Sort frequency
# Convert list of tuples to dictionary value lists

tfidf_dict = defaultdict(list)
for idx, tfidf_num in tfidf_list:
    tfidf_dict[idx].append(tfidf_num)

print('\nTF-IDF for each words: \n')
sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)

# Smallest TF-IDF: words commonly used across all documents and rarely used in the particular document.
print('Smallest 10 tfidf:\n{}\n'.format(sorted_tfidf[:-11:-1]))

# Largest TF-IDF: Term that appears frequently in a particular document, but not often in the corpus.
print('Largest 10 tfidf: \n{}'.format(sorted_tfidf[:11]))
print('-' * 20)

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,  # take all documents into account
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)


print('\n LDA model topics: \n')
for topic, keyword in lda_model.print_topics():
    print('Topic: ', topic)
    print('Keywords: ', keyword)
print('-'*20)

page_lda = lda_model[corpus]

# Compute model perplexity and coherence score
# a measure of how good the model is. lower the better.
print('\nPerplexity and Choerence Score for LDA model:')
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_lda = CoherenceModel(
    model=lda_model, texts=train_words_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)
print('-'*20)

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis

# Build LDA Mallet Model
mallet_path = 'mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(
    mallet_path, corpus=corpus, num_topics=5, id2word=id2word)


print('\n LDA Mallet model topics: \n')
for topic, keyword in ldamallet.show_topics(formatted=False):
    print('Topic: ', topic)
    print('Keywords: ', keyword)
print('-'*20)

coherence_model_ldamallet = CoherenceModel(
    model=ldamallet, texts=train_words_lemmatized, dictionary=id2word, coherence='c_v')

coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# Choerence score lower than lda_model
print('Choerence Score for LDA Mallet model:')
print('\nCoherence Score: ', coherence_ldamallet)
print('-'*20)

# Find the optimal number of topics
def LdaMallet_coherence_values(dictionary, corpus, texts, limit, start = 2, step = 3):
    '''
    Compute c_v coherence for various number of topics
    
    Parameters:
    ---------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    texts: list of input texts
    limit: max num of topics
    
    Returns:
    ---------
    model_list: list of LDA topic models
    coherence_values: corresponding to the LDA model
    '''

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus = corpus, num_topics = num_topics, id2word = dictionary,)
        model_list.append(model)

        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def LDA_coherence_values(dictionary, corpus, texts, limit, chunksize = 100, start=2, step=3):
    '''
    Compute c_v coherence for various number of topics

    Parameters:
    ---------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    texts: list of input text 
    chunksize: number of documents to be used in each training chunk
    limit: max num of topics

    Returns:
    ---------
    model_list: list of LDA topic
    coherence_values: corresponding to the LDA model
    '''

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                                num_topics=num_topics, random_state=100, 
                                                update_every=1, chunksize=100, passes=10, 
                                                alpha='auto', per_word_topics=True)
        
        model_list.append(model)

        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list_mallet, coherence_values_mallet = LdaMallet_coherence_values(dictionary=id2word, corpus=corpus, 
                                                          texts=train_words_lemmatized, start=8, limit=15, step=1)

# show graph
limit = 15
start=8
step=1
x = range(start, limit, step)
plt.plot(x, coherence_values_mallet)
plt.xlabel('Num Topics')
plt.ylabel('Coherence score')
plt.title('Coherence Scores with LDA Mallet Algorith Implementation')
plt.savefig('Topics_Coher_LDA_Mallet')
plt.show()



print('\nLDA Mallet: \n')
for num, cv in zip(x, coherence_values_mallet):
    print('Nun Topics =', num, ' has Coherence Value of', round(cv, 4))
print('-'*20)

model_list, coherence_values = LDA_coherence_values(dictionary=id2word, corpus=corpus, 
                                                          texts=train_words_lemmatized, start=8, limit=15, step=1)

# show graph
limit = 15
start=8
step=1
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel('Num Topics')
plt.ylabel('Coherence score')
plt.title('Coherence Scores with LDA Algorith Implementation')
plt.savefig('Topics_Coher_LDA_Model')
plt.show()

print('\nLDA model: \n')
for num, cv in zip(x, coherence_values):
    print('Nun Topics =', num, ' has Coherence Value of', round(cv, 4))
print('-'*20)

# Usually LDA mallet performs better pick the model with the highest coherence score

optimal_model = model_list_mallet[np.argmax(coherence_values_mallet)]
model_topics = optimal_model.show_topics(formatted = False)

print('\n LDA Mallet topics: \n')
for topic, keyword in optimal_model.print_topics(num_words=10):
    print('Topic: ', topic)
    print('Keywords: ', keyword)
print('-'*20)

# Find the dominant topic in each sentence
# Find the topic number with the highest percentage contributio in that document

def dominant_topic(ldamodel = lda_model, corpus=corpus, texts=body_par):
    # init dataframe
    topics_df = pd.DataFrame()

    # GET MAIN TOPIC IN EACH PARAGRAPH
    # Get throught the pages
    for num, page in enumerate(lda_model[corpus]):
        # Get throught the paragraphs
        for num_par, parag in enumerate(page):
            parag = sorted(parag, key= lambda x: (x[1]), reverse=True)
        
            for j, (topic_num, prop_topic) in enumerate([(5, 0.72200567), (8, 0.27469027)]):
                if j == 0: # => dominant topic
                    # Get list prob. * keywords from the topic
                    pk = lda_model.show_topic(topic_num)
                    topic_keywords = ', '.join([word for word, prop in pk])
                    # Add topic number, probability, keywords and original text to the dataframe
                    topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),
                                                        topic_keywords, texts[num][num_par]]),
                                                        ignore_index=True)
                
                else:
                    break
                
    # Add columns name
    topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Original_Text']

    return topics_df


# data = pd.read_csv('data.csv')

# # converting string to list
# text = data.loc[0, 'Link_1'].strip('][')

# def multi_split(text, seps):
#     '''
#     split string base on multiple separators
#     '''
#     default_sep = seps[0]

#     # make all other seps equal to the default
#     for sep in seps[1:]:
#         text = text.replace(sep, default_sep)

#     return [i.strip() for i in text.split(default_sep)]

# text = multi_split(text, ('", ', "', "))

# # Remove leading ' or "
# for num, i in enumerate(text):
#     text[num] = re.sub('^["\']', '', i)

# link_1 = pd.DataFrame(text, columns=['text'])

# # Counting vocabulary of words
# link_1.shape  # num of paragraphs

# # Delete paragraphs with 0 characters
# link_1 = link_1.loc[link_1['text'].str.len() > 0, :]
# link_1['text'].str.len().max()  # 704
# link_1['text'].str.len().min()  # 2

# # number of tokens for each string
# link_1['text'].str.split().str.len().max()  # 118

# # print string with digits
# for i in range(digit.shape[0]):
#     print('\n\n', digit.iloc[i, 0])

# # unique tokens
# set(link_1.loc[0, 'text'].split())

# # Frequency of words in all the text
# dist = nltk.FreqDist(' '.join(text).split())
# len(dist)  # number of unique tokens in all the text >> 1384

# # Sort frequency
# sorted(dist.items(), key=lambda x: x[1], reverse=True)

# vocabl = dist.keys()
# freqwords = [w for w in vocabl if len(w) > 3 and dist[w] > 20]
# # TODO: code product Siri, Alexa, Google Assistant names as <product_name>
# # TODO: code companies Apple, Google, Amazon, names as <company_name>

# # remove stopwords
# # Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
# stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
#              "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
#              "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
#              "during", "each", "few", "for", "from", "further", "had", "has", "have",
#              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
#              "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
#              "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me",
#              "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
#              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
#              "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
#              "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
#              "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
#              "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
#              "we're", "we've", "were", "what", "what's", "when", "when's", "where",
#              "where's", "which", "while", "who", "who's", "whom", "why", "why's",
#              "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
#              "yours", "yourself", "yourselves", "however"]

# for num, paragraph in enumerate(text):
#     text[num] = ' '.join(
#         [w for w in paragraph.strip().split() if not w in stopwords])

# # Normalization and stemming (going back to root)
# porter = nltk.PorterStemmer()
# stemmed_text = [porter.stem(t) for t in ' '.join(text).split()]

# # number of unique stemmed tokens in all the text >> 1189
# stemmed_dist = nltk.FreqDist(stemmed_text)
# len(stemmed_dist)
# [w for w in stemmed_dist.keys() if len(w) > 3 and stemmed_dist[w] > 10]
# sorted(stemmed_dist.items(), key=lambda x: x[1], reverse=True)

# # Lemmatization: Stemming, but resulting stems are all valid words
# # nltk.download('wordnet')
# WNlemma = nltk.WordNetLemmatizer()
# lemma_text = [WNlemma.lemmatize(t) for t in ' '.join(text).split()]

# # number of unique lemmatize tokens in all the text >> 1335
# len(nltk.FreqDist(lemma_text))

# # Tokenization
# # nltk.download('punkt')
# # Tokenize a string to split off punctuation other than periods
# nltk.word_tokenize(' '.join(text))

# # create sentences: change grain of the analysis
# sentences = nltk.sent_tokenize(' '.join(text))
# # TODO: add dot at the end of title (headline)
# len(sentences)

# # TF - IDF
# # min document frequency of 5
# vect = TfidfVectorizer(min_df=5).fit(text)
# feature_names = np.array(vect.get_feature_names())

# text_vectorized = vect.transform(text)

# sorted_tfidf_index = text_vectorized.max(0).toarray()[0].argsort()

# # Smallest: words commonly used across all documents and rarely used in the particular document.
# print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
# # Largest: Term that appears frequently in a particular document, but not often in the corpus.
# # "paragraph hedline - topics"
# print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

# # TOPIC MODELLING
# # Use CountVectorizor to find three letter tokens, remove stop_words,
# # remove tokens that don't appear in at least 5 documents,
# # remove tokens that appear in more than 30% of the documents
# vect = CountVectorizer(min_df=5, max_df=0.3, stop_words='english',
#                        token_pattern='(?u)\\b\\w\\w\\w+\\b')

# # Fit and transform
# X = vect.fit_transform(text)

# # Convert sparse matrix to gensim corpus
# corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# # Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
# id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# # Estimate LDA model parameters on the corpus
# ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=3, id2word=id_map,
#                                            passes=25, random_state=34)

# # Find list of topics and the most significant 10 words

# def lda_topics():
#     return ldamodel.print_topics()

# # topic_distribution in a new document
# new_doc = list(text[89])
# new_doc_vec = vect.transform(new_doc)
# new_corpus = gensim.matutils.Sparse2Corpus(
#     new_doc_vec, documents_columns=False)
# list(ldamodel.get_document_topics(new_corpus))[0]

# # topic_names

# def topic_names():

#     # List of potential topics

#     topic_label = ['Health', 'Science', 'Automobiles', 'Politics', 'Government', 'Travel',
#                    'Computers & IT', 'Sports', 'Business', 'Society & Lifestyle', 'Religion', 'Education']
#     topics = lda_topics()
#     result = []
#     for _, dist in topics:
#         similarity = []
#         for topic in topic_label:
#             similarity.append(document_path_similarity(dist, topic))
#         # associate similarity with the labels using zip
#         best_topic = sorted(zip(similarity, topic_label))[-1][1]

#         result.append(best_topic)
#     return result

# # INFORAMTION EXTRACTION
# # N-GRAM add context by ading sequences of word.

# # POS (Part-of-speech) tagging >> Information Extraction
# # if you're interested in extracting specific tags (word classes / types)
# # nltk.download('tagsets')
# nltk.help.upenn_tagset('')  # help tags meaning
# # Document Similarity
# # nltk.download('averaged_perceptron_tagger')

# def convert_tag(tag):
#     '''
#     Convert tag given by nltk.pos_tag to the wordnet.synsets'tag
#     '''

#     # n = noun, v = verb, a = adjective, r = adverb
#     tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}

#     # not all tags are being converted
#     try:
#         return tag_dict[tag[0]]
#     except KeyError:
#         return None

#     tag_dict['N']

# def doc_to_synsets(doc):
#     '''
#     Retuns a list of synsets in document
#     '''

#     token = nltk.word_tokenize(doc)
#     word_tag = nltk.pos_tag(token)

#     synsets = []

#     for word, tag in word_tag:
#         tag = convert_tag(tag)
#         # Synset simple interface in nltk to look up words in WordNet
#         # synset instance are grupings of synonymous words
#         synset = wn.synsets(word, pos=tag)
#         if len(synset) != 0:
#             synsets.append(synset[0])
#         else:
#             continue

#     return synsets

# def similarity_score(s1, s2):
#     '''
#     Calculate the normalized similarity score.
#     Input: s1, s2, list of synsets from doc_to_synsets
#     Output: normalized similarity score of s1 onto s2
#     '''

#     largest_sim_values = []
#     # similarity score
#     for syn1 in s1:
#         similarity_values = []
#         for syn2 in s2:
#             sim_val = wn.path_similarity(syn1, syn2)
#             if sim_val is not None:
#                 similarity_values.append(sim_val)
#         if len(similarity_values) != 0:
#             largest_sim_values.append(max(similarity_values))

#     return sum(largest_sim_values) / len(largest_sim_values)

# s1 = doc_to_synsets(text[0])
# s2 = doc_to_synsets(text[1])

# similarity_score(s1, s2)

# def document_path_similarity(doc1, doc2):
#     '''
#     Finds symmetrical similarity between doc1 and doc2
#     '''

#     synsets1 = doc_to_synsets(doc1)
#     synsets2 = doc_to_synsets(doc2)

#     return (similarity_score(synsets1, synsets2) +
#             similarity_score(synsets2, synsets1)) / 2
