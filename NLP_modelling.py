import logging
import os
import time
import warnings

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.matutils import corpus2csc
from gensim.models import CoherenceModel, LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('preprocessed_comments50k.csv', index_col=0)

# Create a new columns with tokens
comments = df['clean_body'].map(lambda x: str(x).strip().split())

# Create Dictionary
dictionary = corpora.Dictionary(comments)
# Check if a word exist
# [k for k, v in dictionary.items() if v == 'trust']

# Verify deleted words
# tmp = dict(dictionary)
# len(dictionary)

# Filter out words that occur less than 2 comments, or more than 90% of comments
dictionary.filter_extremes(no_below=2, no_above=0.8) 
# words very sparse
# NOTE: consider to increase the number of comments
#len(dictionary)
#[v for v in tmp.values() if v not in dictionary.values()]

# Term Document Frequency >> (id, freq) for each page
corpus = [dictionary.doc2bow(text) for text in comments]
#[(dictionary[k], f) for k, f in corpus[0]]

# Words used in how many comments?
# unique_words_comment = [dictionary[w[0]] for k in corpus for w in k]
# unique_words_comment = pd.DataFrame(unique_words_comment)
# unique_words_comment[0].value_counts().describe()
# tmp = unique_words_comment[0].value_counts()
# tmp.to_csv('tot_50k_dict.csv', header=None, index=True)

## MODELS
# Download mallet software and run following commands on powershell
# wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip -OutFile mallet-2.0.8.zip # * updated if needed *
# then unzip

# Logging Gensim's output
log_file = os.getcwd() + r'\logging' + r'\log_%s.txt' % int(time.time())

logging.basicConfig(filename=log_file,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Find the optimal number of topics


def LdaMallet_topics(dictionary, corpus, texts, limit, start, step):
    '''
    
    Parameters:
    ---------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    texts: list of input texts
    limit: max num of topics
    
    Returns:
    ---------
    model_list: list of LDA Mallet topic models
    '''

    os.environ['MALLET_HOME'] = r'C:\\mallet-2.0.8\\'
    mallet_path = r'C:\\mallet-2.0.8\\bin\\mallet'
    model_list = []

    for num_topics in range(start, limit, step):
        print('Running model with number of topics: ', num_topics)
        model = LdaMallet(mallet_path, corpus = corpus, 
                          num_topics = num_topics, id2word = dictionary, 
                          random_seed=123)

        model_list.append(model)

    return model_list

# Mallet_model_list = LdaMallet_topics(dictionary=dictionary, corpus=corpus, 
#                                                           texts=comments, start=5, limit=200, step=5)
# SAVE MODELS
# for i, num in zip(Mallet_model_list, range(5, 200, 5)):
#     tmp_file = datapath('LdaMallet_model\\model{:d}'.format(num))
#     i.save(tmp_file)

# SAVE DICTIONARY
tmp_file = datapath('dictionary50mi')
dictionary.save(tmp_file)


def LdaGensim_topics(dictionary, corpus, texts, limit, start, step, alpha, beta):
    '''
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
    '''

    model_1_001 = []
    model_10_01 = []
    model_50_05 = []

    passage = 0
    for a, b in zip(alpha, beta):
        for num_topics in range(start, limit, step):
            print('Running model with number of topics: ', num_topics)
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                                    num_topics=num_topics, random_state=123, 
                                                    update_every=1, chunksize=50000, passes=5, 
                                                    alpha=[a]*num_topics, eta=b,  per_word_topics=True)
            
            if passage == 0:
                print('Saving model with alpha=1/k and beta=0.01')
                model_1_001.append(model)
                
            if passage == 1:
                print('Saving model with alpha=10/k and beta=0.1')
                model_10_01.append(model)

            if passage == 2:
                print('Saving model with alpha=50/k and beta=0.5')
                model_50_05.append(model)
        
        passage +=1
        print('\n')

    return model_1_001, model_10_01, model_50_05

model_1_001, model_10_01, model_50_05 = LdaGensim_topics(dictionary=dictionary, corpus=corpus, 
                                                            texts=comments, start=5, limit=501, step=5, alpha = [1.00, 10.00, 50.00], beta = [0.01, 0.1, 0.5])



# SAVE MODELS
for i, num in zip(model_1_001, range(5, 501, 5)):
    tmp_file = datapath('model_1_001\\{:d}'.format(num))
    i.save(tmp_file)

for i, num in zip(model_10_01, range(5, 501, 5)):
    tmp_file = datapath('model_10_01\\{:d}'.format(num))
    i.save(tmp_file)

for i, num in zip(model_50_05, range(5, 501, 5)):
    tmp_file = datapath('model_50_05\\{:d}'.format(num))
    i.save(tmp_file)


#NOTE: multicores implementation
def LdaGensim_topics_one_model(dictionary, corpus, texts, limit, start, step, alpha, beta):
    '''
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
    '''

    model_01_0001 = []

    for num_topics in range(start, limit, step):
        print('Running model with number of topics: ', num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                                num_topics=num_topics, random_state=123, 
                                                update_every=1, chunksize=50000, passes=5, 
                                                alpha=[alpha]*num_topics, eta=beta,  per_word_topics=True)
        
        print('Saving model with alpha=0.1/k and beta=0.001')
        model_01_0001.append(model)


    return model_01_0001

model_01_0001 = LdaGensim_topics_one_model(dictionary=dictionary, corpus=corpus, 
                                                            texts=comments, start=5, limit=501, step=5, alpha = 0.1, beta = 0.001)

for i, num in zip(model_01_0001, range(5, 501, 5)):
    tmp_file = datapath('model_01_0001\\{:d}'.format(num))
    i.save(tmp_file)