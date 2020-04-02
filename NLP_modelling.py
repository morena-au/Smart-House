import logging
import os
import time
import warnings
import re
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.matutils import corpus2csc, Sparse2Corpus
from gensim.models import CoherenceModel, LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
import NLP_visualization as NLP_vis
import MySQL_data as data
import clean_text as clean_fun
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('.\\DataSource_backup\\sub_onetree.csv', encoding="utf-8")
NLP_vis.words_count(df["clean_text"])

# get source comments for further investigations
# comments = data.comments

# remove rows with less than 15 words (short observations)
df = df.loc[df['clean_text'].map(lambda x: len(str(x).strip().split())) > 15,]

# Divide the data in 80% training and 20% test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

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
dictionary.filter_extremes(no_below=5, no_above=0.4) 
NLP_vis.dictionary_freq_words(dictionary, False, 30)


# ## MODELS
# # Download mallet software and run following commands on powershell
# # wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip -OutFile mallet-2.0.8.zip # * updated if needed *
# # then unzip

# # Logging Gensim's output
# log_file = os.getcwd() + r'\logging' + r'\log_%s.txt' % int(time.time())

# logging.basicConfig(filename=log_file,
#                     format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)

# # Find the optimal number of topics


# def LdaMallet_topics(dictionary, corpus, texts, limit, start, step):
#     '''
    
#     Parameters:
#     ---------
#     dictionary: Gensim dictionary
#     corpus: Gensim corpus
#     texts: list of input texts
#     limit: max num of topics
    
#     Returns:
#     ---------
#     model_list: list of LDA Mallet topic models
#     '''

#     os.environ['MALLET_HOME'] = r'C:\\mallet-2.0.8\\'
#     mallet_path = r'C:\\mallet-2.0.8\\bin\\mallet'
#     model_list = []

#     for num_topics in range(start, limit, step):
#         print('Running model with number of topics: ', num_topics)
#         model = LdaMallet(mallet_path, corpus = corpus, 
#                           num_topics = num_topics, id2word = dictionary, 
#                           random_seed=123)

#         model_list.append(model)

#     return model_list

# # Mallet_model_list = LdaMallet_topics(dictionary=dictionary, corpus=corpus, 
# #                                                           texts=comments, start=5, limit=200, step=5)
# # SAVE MODELS
# # for i, num in zip(Mallet_model_list, range(5, 200, 5)):
# #     tmp_file = datapath('LdaMallet_model\\model{:d}'.format(num))
# #     i.save(tmp_file)

# # SAVE DICTIONARY
# tmp_file = datapath('dictionary50mi')
# dictionary.save(tmp_file)


# def LdaGensim_topics(dictionary, corpus, texts, limit, start, step, alpha, beta):
#     '''
#     Parameters:
#     ---------
#     dictionary: Gensim dictionary
#     corpus: Gensim corpus
#     texts: list of input text 
#     chunksize: number of documents to be used in each training chunk
#     limit: max num of topics

#     Returns:
#     ---------
#     model_list: list of LDA topic
#     '''

#     model_1_001 = []
#     model_10_01 = []
#     model_50_05 = []

#     passage = 0
#     for a, b in zip(alpha, beta):
#         for num_topics in range(start, limit, step):
#             print('Running model with number of topics: ', num_topics)
#             model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
#                                                     num_topics=num_topics, random_state=123, 
#                                                     update_every=1, chunksize=50000, passes=5, 
#                                                     alpha=[a]*num_topics, eta=b,  per_word_topics=True)
            
#             if passage == 0:
#                 print('Saving model with alpha=1/k and beta=0.01')
#                 model_1_001.append(model)
                
#             if passage == 1:
#                 print('Saving model with alpha=10/k and beta=0.1')
#                 model_10_01.append(model)

#             if passage == 2:
#                 print('Saving model with alpha=50/k and beta=0.5')
#                 model_50_05.append(model)
        
#         passage +=1
#         print('\n')

#     return model_1_001, model_10_01, model_50_05

# model_1_001, model_10_01, model_50_05 = LdaGensim_topics(dictionary=dictionary, corpus=corpus, 
#                                                             texts=comments, start=5, limit=501, step=5, alpha = [1.00, 10.00, 50.00], beta = [0.01, 0.1, 0.5])



# # SAVE MODELS
# for i, num in zip(model_1_001, range(5, 501, 5)):
#     tmp_file = datapath('model_1_001\\{:d}'.format(num))
#     i.save(tmp_file)

# for i, num in zip(model_10_01, range(5, 501, 5)):
#     tmp_file = datapath('model_10_01\\{:d}'.format(num))
#     i.save(tmp_file)

# for i, num in zip(model_50_05, range(5, 501, 5)):
#     tmp_file = datapath('model_50_05\\{:d}'.format(num))
#     i.save(tmp_file)


# #NOTE: multicores implementation
# def LdaGensim_topics_one_model(dictionary, corpus, texts, limit, start, step, alpha, beta):
#     '''
#     Parameters:
#     ---------
#     dictionary: Gensim dictionary
#     corpus: Gensim corpus
#     texts: list of input text 
#     chunksize: number of documents to be used in each training chunk
#     limit: max num of topics

#     Returns:
#     ---------
#     model_list: list of LDA topic
#     '''

#     model_01_0001 = []

#     for num_topics in range(start, limit, step):
#         print('Running model with number of topics: ', num_topics)
#         model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
#                                                 num_topics=num_topics, random_state=123, 
#                                                 update_every=1, chunksize=50000, passes=5, 
#                                                 alpha=[alpha]*num_topics, eta=beta,  per_word_topics=True)
        
#         print('Saving model with alpha=0.1/k and beta=0.001')
#         model_01_0001.append(model)


#     return model_01_0001

# model_01_0001 = LdaGensim_topics_one_model(dictionary=dictionary, corpus=corpus, 
#                                                             texts=comments, start=5, limit=501, step=5, alpha = 0.1, beta = 0.001)

# for i, num in zip(model_01_0001, range(5, 501, 5)):
#     tmp_file = datapath('model_01_0001\\{:d}'.format(num))
#     i.save(tmp_file)