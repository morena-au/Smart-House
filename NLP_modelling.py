import logging
import os
import time
import warnings
import copy
import re
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.matutils import corpus2csc, Sparse2Corpus
from gensim.models import LdaModel, LdaMulticore
#from gensim.models.wrappers import LdaMallet
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
import NLP_visualization as NLP_vis
import MySQL_data as data
import clean_text as clean_fun
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df_train = pd.read_csv('.\\DataSource_backup\\sub_onetree_train.csv', encoding="utf-8")
NLP_vis.words_count(df_train["clean_text"])

# get source comments for further investigations
# comments = data.comments

# Train Bigram Models
# ignore terms that appeared in less than 2 documents
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)

X = bigram_vectorizer.fit_transform(df_train['clean_text'].tolist())

term2id = bigram_vectorizer.vocabulary_

# get gensim dictionary
# https://stackoverflow.com/questions/21552518/using-scikit-learn-vectorizers-and-vocabularies-with-gensim
# transform sparse matrix into gensim corpus: Term Document Frequency (id, freq) for each text
corpus = Sparse2Corpus(X, documents_columns=False)
dictionary = corpora.Dictionary.from_corpus(corpus, id2word= {v:k for (k, v) in term2id.items()})

# Words used in how many texts?
NLP_vis.vocabulary_descriptive(dictionary, corpus)

# Filter out words that occur less than 5 comments, or more than 80% of comments
filter_dict = copy.deepcopy(dictionary)
filter_dict.filter_extremes(no_below=5, no_above=0.4) 
NLP_vis.vocabulary_freq_words(filter_dict, False, 30)

# SAVE DICTIONARY
tmp_file = datapath('vocabulary\\nb5_na04')
filter_dict.save(tmp_file)

#Update corpus to the new dictionary
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict.items()})

train_bigram = bigram_vectorizer.fit(df_train['clean_text'].tolist())

# SAVE BIGRAM
filename = datapath('train_bigram\\nb5_na04_bigram.pkl')
with open(filename, "wb") as f:
    pickle.dump(train_bigram, f)

X = bigram_vectorizer.transform(df_train['clean_text'].tolist())

corpus = Sparse2Corpus(X, documents_columns=False)
NLP_vis.vocabulary_descriptive(filter_dict, corpus)

## MODELS

# Logging Gensim's output
# return time in seconds since the epoch
log_file = os.getcwd() + r'\logging' + r'\log_%s.txt' % int(time.time())

logging.basicConfig(filename=log_file,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def LdaGensim_topics(dictionary, corpus, limit, start, step, alpha, beta):
    '''
    Parameters:
    ---------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    limit: max num of topics (end at limit -1)
    start: min number of topics
    step: increment between each integer
    alpha: list of prior on the per-document topic distribution.
    eta: list of prior on the per-topic word distribution. 

    Returns:
    ---------
    model_list: list of LDA topic
    '''

    # initialize an empty dictionary
    models = {}

    for a in alpha: 
        print('Running model with alpha={}/k'.format(a))
        for b in beta:
            print('Running model with beta={}'.format(b))
            for num_topics in range(start, limit, step):
                print('Running model with number of topics (k): ', num_topics)
                model = LdaModel(corpus=corpus, id2word=dictionary,
                                     num_topics=num_topics, random_state=123, 
                                     passes=10, 
                                     alpha=[a]*num_topics, eta=b,
                                     per_word_topics=True)
                
                name = "a{0}_b{1}_k{2}".format(a, b, num_topics)
                models[name] = model

                print('\n')

    return models

models = LdaGensim_topics(dictionary=filter_dict, corpus=corpus, 
                                                         start=5, limit=201, step=5, 
                                                         alpha = [0.01, 0.1, 1, 10], 
                                                         beta = [0.01, 0.1, 1, 10])


# [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50] triggered insufficient memory on RDP server.
#time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1585905944))

# SAVE MODELS
# Write pickles incrementally to a file 
# One pickles equal to a combination of alpha beta across all number of topics)
# 16 (alpha x beta) pickles with 40 dictionary inside (a topic num for each dict) 
# https://stackoverflow.com/questions/37954324/how-to-load-one-line-at-a-time-from-a-pickle-file
# filename = datapath('train_models\\nb5_na04_models.sav')
# joblib.dump(models, filename)


## MemoryError
# for k, v in models.items():
#     tmp_file = datapath('train_models\\nb5_na04\\{}'.format(k))
#     v.save(tmp_file)


