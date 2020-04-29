import re
import os
import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

# function to plot most frequent terms
def freq_words(x, ascending=False, terms = 30):
    """
    Plot word frequency
    Input: pd.Series, direction, num. of words
    """
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top most frequent words
    d = words_df.sort_values("count", ascending=ascending)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d[:terms], x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.title("Words Frequency")
    plt.xticks(rotation=45)
    plt.show()

# Plot words count distribution across texts
def words_count(x):
    """
    Words count distribution 
    Input: pd.Series
    """
    word_dist = [len(word.strip().split()) for word in x]
    plt.figure(figsize=(16,7), dpi=160)
    plt.hist(word_dist, bins = 500, color='navy')
    plt.text(1000, 450, "Mean   : " + str(round(np.mean(word_dist))))
    plt.text(1000, 400, "Median : " + str(round(np.median(word_dist))))
    plt.text(1000, 350, "Stdev   : " + str(round(np.std(word_dist))))
    plt.text(1000, 300, "1%ile    : " + str(round(np.quantile(word_dist, q=0.01))))
    plt.text(1000,  250, "99%ile  : " + str(round(np.quantile(word_dist, q=0.99))))
    plt.gca().set(xlim=(0, max(word_dist)), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,max(word_dist),18))
    plt.title('Word Counts Distribution', fontdict=dict(size=22))
    plt.show()

def vocabulary_descriptive(dictionary, corpus):
    """
    Input: gensim dictionary and corpus
    Output: Descriptive statistics: word used in how many texts
    """
    unique_words_text = pd.DataFrame.from_dict(dictionary.dfs, orient="index")
    return print(unique_words_text[0].describe())

# function to plot most frequent terms
def vocabulary_freq_words(dictionary, ascending=False, terms = 30):
    """
    Plot word frequency by the number of documents that contains it
    Input: gensim dictionary, direction, num. of words
    """
    
    dict_id2token = {id_:tok_ for (tok_, id_) in dictionary.token2id.items()}
    dict_token2freq = {dict_id2token[k]:v for (k,v) in dictionary.dfs.items()}

    words_df = pd.DataFrame.from_dict(dict_token2freq, orient="index", columns=["count"])

    # selecting top most frequent words
    d = words_df.sort_values("count", ascending=ascending).reset_index()
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d[:terms], x= "index", y = "count")
    ax.set(ylabel = 'Count')
    plt.title("Words Frequency")
    plt.xticks(rotation=45)
    plt.show()

