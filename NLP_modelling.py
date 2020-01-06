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
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('preprocessed_comments.csv')

# Train Bigram and Trigram Models
# higher threshold fewer phrases.
comment_token_list = df['clean_body'].map(lambda x: str(x).strip().split())

bigram = gensim.models.Phrases(comment_token_list, min_count=70, threshold=15)
trigram = gensim.models.Phrases(bigram[comment_token_list], min_count=70, threshold=15)

dist = nltk.FreqDist(
    [word for comment in trigram[bigram[comment_token_list]] for word in comment if '_' in word])

# Sort frequency
print('Sorted trigrams: \n')
print(sorted(dist.items(), key=lambda x: x[1], reverse=True))
print('-'*20)

df['clean_body'] = list(trigram[bigram[comment_token_list]])
df['clean_body'] = df['clean_body'].map(lambda x: ' '.join(x))

# function to plot most frequent terms
def freq_words(x, ascending=False, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = nltk.FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top most frequent words
  d = words_df.sort_values("count", ascending=ascending)
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d[:terms], x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.xticks(rotation=45)
  plt.show()

freq_words(df['clean_body'], True, 50)

# Create a new columns with tokens
comments = df['clean_body'].map(lambda x: str(x).strip().split())

# Create Dictionary
dictionary = corpora.Dictionary(comments)
# Check if a word exist
# [k for k, v in dictionary.items() if v == 'trust']

# Verify deleted words
#tmp = dict(dictionary)
#len(dictionary)

# Filter out words that occur less than 2 comments, or more than 90% of comments
dictionary.filter_extremes(no_below=2, no_above=0.9) 
# words very sparse, no_above 0.20 for digit and use
# NOTE: consider to increase the number of comments
#len(dictionary)
#[v for v in tmp.values() if v not in dictionary.values()]

# Term Document Frequency >> (id, freq) for each page
corpus = [dictionary.doc2bow(text) for text in comments]
#[(dictionary[k], f) for k, f in corpus[0]]

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


def LdaMallet_coherence_values(dictionary, corpus, texts, limit, start, step):
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

    os.environ['MALLET_HOME'] = r'C:\\mallet-2.0.8\\'
    mallet_path = r'C:\\mallet-2.0.8\\bin\\mallet'
    model_list = []
    coherence_values = []

    for num_topics in range(start, limit, step):
        print('Running model with number of topics: ', num_topics)
        model = LdaMallet(mallet_path, corpus = corpus, 
                          num_topics = num_topics, id2word = dictionary)

        model_list.append(model)

        coherencemodel = CoherenceModel(model = model, texts = texts,
                                        dictionary = dictionary, coherence = 'c_v')

        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

Mallet_model_list, Mallet_coherence_values = LdaMallet_coherence_values(dictionary=dictionary, corpus=corpus, 
                                                          texts=comments, start=5, limit=200, step=5)
# SAVE
for i, num in zip(Mallet_model_list, range(5, 200, 5)):
    tmp_file = datapath('train_model\\model{:d}'.format(num))
    i.save(tmp_file)

coherence_gensim_c_v = Mallet_coherence_values

cao_juan_2009 =[]
arun_2010 =[]
coherence_mimno_2011 = []


for i in range(len(Mallet_model_list)):
    cao_juan_2009.append(evaluate.metric_cao_juan_2009(Mallet_model_list[i].get_topics()))

    arun_2010.append(evaluate.metric_arun_2010(Mallet_model_list[i].get_topics(),  
                        np.array([x.transpose()[1] for x in np.array(list(Mallet_model_list[i].load_document_topics()))]),
                        np.array([len(x) for x in comments])))

    coherence_mimno_2011.append(evaluate.metric_coherence_mimno_2011(Mallet_model_list[i].get_topics(), 
                                                                corpus2csc(corpus).transpose(), return_mean=True))

# Write evaluation metrics
pd.DataFrame(coherence_gensim_c_v).to_csv('coherence_gensim_c_v.csv', header=False)
pd.DataFrame(cao_juan_2009).to_csv('cao_juan_2009.csv', header=False)
pd.DataFrame(arun_2010).to_csv('arun_2010.csv', header=False)
pd.DataFrame(coherence_mimno_2011).to_csv('coherence_mimno_2011.csv', header=False)
