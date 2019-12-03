import json
import re
import warnings
from collections import Counter, defaultdict

import gensim
import gensim.corpora as corpora
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
import spacy
from collections import OrderedDict
from bokeh.io import output_notebook
from bokeh.models import Label
from bokeh.plotting import figure, output_file, show
from gensim.models import CoherenceModel, LsiModel, HdpModel, LdaModel
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from wordcloud import STOPWORDS, WordCloud

## Run in terminal with the size wanted 
# python3 pushshift_comments.py 10

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import training data

df = pd.read_csv('comments.csv', encoding='utf8')

# Cleaning up the comments
# nltk.download('stopwords')  # (run python console)
# nltk.download('wordnet')  # (run python console)
# python3 -m spacy download en  # (run in terminal)

# Extracting URLs: external links often informative, but they add unwanted noise to our NLP model
# Strip out hyperlinks and copy thme in a new column URL

# Find URL
def find_URL(comment):
    return re.findall(r'((?:https?:\/\/(?:www\.)?|(?:pic\.|www\.)(?:\S*\.))(?:\S*\/))', comment)

# apply the function on the body column
df['URL'] = df.body.apply(find_URL)

# create a colummn with pre-processed try:
df['clean_body'] = [re.sub(r"((?:https?:\/\/(?:www\.)?|(?:pic\.|www\.)(?:\S*\.))(?:\S*\/))",'', x) for x in df['body']]

# Find all internal hyphen words and consider them as full words (technical vocabulary)
df['clean_body'] = [re.sub(r'([a-zA-Z]+)(-)([a-zA-Z]+)', r'\g<1>\g<3>', x) for x in df['clean_body']]

# Standardizing the informal language of comments
# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['etc', 'however', 'there', 'home', 'week', 'also', 'like'])

#We specify the stemmer or lemmatizer we want to use
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
wordnet_lemmatizer = WordNetLemmatizer()

def clean_comment(comment, bigrams=False, lemma=False, allowed_postags=['NOUN', 'ADJ', 'VERB']):
    comment = comment.lower() # ? consider to make general the name of companies or decives
    comment = re.sub('&gt', ' ', comment) # remove all copied text into a comment '&gt'
    comment = re.sub('[^\s\w]', ' ', comment) # strip out everything (punctuation) that is not Unicode whitespace or word character
    comment = re.sub('[0-9]+', ' ', comment) # remove digits

    # remove stop_words
    comment_token_list = [word for word in comment.strip().split() if word not in stop_words]
    
    # remove one character word
    comment_token_list = [word for word in comment_token_list if len(word) > 1]
    
    # keeps word meaning
    if lemma == True:
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        nlp = spacy.load('en', disable=['parser', 'ner'])
        # https://spacy.io/api/annotation
        comment_text = nlp(' '.join(comment_token_list))
        comment_token_list = [token.lemma_ for token in comment_text if token.pos_ in allowed_postags]
        #comment_token_list = [wordnet_lemmatizer.lemmatize(word) for word in comment_token_list]
    
    # harsh to the root of the word
    else:
        comment_token_list = [word_rooter(word) for word in comment_token_list]

    if bigrams:
        comment_token_list = bigram[comment_token_list]
    
    comment = ' '.join(comment_token_list)
    
    return comment

# Apply function to clean the comment
df['clean_body'] = df.clean_body.apply(clean_comment, lemma=True)

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

freq_words(df['clean_body'], True, 20)

def comment2token(comments):
    '''
    Return sequence (stream, generator) of sentences,
    with each sentence a list of tokens
    '''
    return [comment.split() for comment in comments]

comment_token_list = comment2token(df['clean_body'])

# Train Bigram and Trigram Models
# higher threshold fewer phrases.
bigram = gensim.models.Phrases(comment_token_list, min_count=3, threshold=20)

# Frequency of n-gram words
dist = nltk.FreqDist(
    [word for comment in bigram[comment_token_list] for word in comment if '_' in word])

# Sort frequency
print('Sorted trigrams: \n')
print(sorted(dist.items(), key=lambda x: x[1], reverse=True))
print('-'*20)


# df['clean_body'] = df.clean_body.apply(clean_comment, bigrams=True, lemma=True)

# FURTEHR CONSIDERATION
# Bot comments - duplicate

# Create a new columns with tokens
df['token_text'] = [[word for word in comment.split()] for comment in df['clean_body']]

# Add Wikipedia corpus
with open('train_wiki.json', 'r') as f:
    train_wiki = json.load(f)

link = []
category = []
body_par = []

for item in train_wiki:
    link.append(item['link'])
    category.append(item['category'])
    body_par.append(item['body_par'])

wiki_text= []
for elm in body_par:
    wiki_text.append(' '.join(elm))

wiki_text = pd.DataFrame({'wiki_text': wiki_text})

# Apply function to clean the comment
wiki_text['wiki_clean'] = wiki_text.wiki_text.apply(clean_comment, lemma=True)
wiki_text['token_text'] = [[word for word in comment.split()] for comment in wiki_text['wiki_clean']]

comment = pd.concat([wiki_text['token_text'], df['token_text']], ignore_index=True)
# comment = df['token_text']

# Create Dictionary
dictionary = corpora.Dictionary(comment)

# Term Document Frequency >> (id, freq) for each page
corpus = [dictionary.doc2bow(text) for text in comment]

print('\nPrint words and frequencies in the first comment:\n')
print([[(dictionary[id], freq) for id, freq in page] for page in corpus[:1]])
print('-' * 20)

# TF-IDF model as pre-processing step
cv = CountVectorizer()
comment_token_vector = cv.fit_transform(df['clean_body'])

# Compute the IDF values
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(comment_token_vector)

# Compute the TFIDF score (new unseen dataset)
count_vector = cv.transform(df['clean_body'])
tfidf_vector = tfidf_transformer.transform(count_vector)

# Check if it makes sense
feature_names = np.array(cv.get_feature_names())

# max(0) equal max by columns, argsort: return the indices that would sort an array
sorted_tfidf_index = tfidf_vector.max(0).toarray()[0].argsort()

# Smallest: words commonly used across all documents and rarely used in the particular document.
print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))

print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

## MODELS
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
    
    mallet_path = 'mallet-2.0.8/bin/mallet'
    
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        print('Running model with number of topics: ', num_topics)
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus = corpus, num_topics = num_topics, id2word = dictionary)
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
        print('Running model with number of topics: ', num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                                num_topics=num_topics, random_state=100, 
                                                update_every=1, chunksize=100, passes=10, 
                                                alpha='auto', per_word_topics=True)
        
        model_list.append(model)

        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = LDA_coherence_values(dictionary=dictionary, corpus=corpus, 
                                                          texts=comment, start=5, limit=30, step=1)


limit = 30
start=5
step=1
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.plot(x[np.argmax(coherence_values)], max(coherence_values), 'or')
plt.text(x[np.argmax(coherence_values)]+0.2, max(coherence_values), 
         r'({}, {})'.format(x[np.argmax(coherence_values)], np.round(max(coherence_values), 2)))
plt.xlabel('Num Topics')
plt.ylabel('Coherence score')
plt.title('Coherence Scores with LDA Algorith Implementation')
# plt.savefig('output/Topics_Coher_LDA_Model_page')
plt.show()

print('\nLDA model: \n')
for num, cv in zip(x, coherence_values):
    print('Nun Topics =', num, ' has Coherence Value of', round(cv, 4))
print('-'*20)

# Extract the model base on the topic number
def model_extract(num_topics):
    '''
    Input: num_topics (int)
    '''
    # get number of topics
    num, _ = zip(*list(zip(x,coherence_values)))
    model = model_list[[i for i in range(len(num)) if num[i] == num_topics][0]]
    
    return model

model = model_extract(10)

# save model to disk
# temp_file = datapath("TrainedModel")
# model.save(temp_file)
# Load a potentially pretrained model from disk.
# lda = LdaModel.load(temp_file)

model_topics = model.show_topics(num_topics=10, num_words=10,formatted = False)

print('\n LDA topics: \n')
for topic, keyword in model.print_topics(num_words=10):
    print('Topic: ', topic)
    print('Keywords: ', keyword)
print('-'*20)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, corpus, dictionary, mds='TSNE')
vis

# INTERPRETATION:
# Output: prop mass function over the words in the model for each topic.
# Bar: list top 35 words given the topic
# Red bars: frequency of each words given a topic
# Gray bars: overall word frequency

# # EASY TO INTERFER MEANING
# re-ranch and introduce new words that are specific to the topic using the 
# parameter lambda >> decreasing lambda more weight on the ratio red to gray
# (freq given the topic to the overall frequency) jargon more lament (lament improve
# readability for those who are not familiar with the topic)

# Visualize the unexplained portion of a words within a topic by simply
# going with the mouse over the word. >> words used in different topics 
# (conditional topic distribution for a given word) >> how other topics use the word <<

# Distance between topics in the scatter plot is an approximation of the
# difference between topic distribution. >> approximation of the sematic relationship

# Bubble size is the topic prevalence
# Indices inside the bubble is the sorted order by area >> num 1 is the most popular topic to least
# Distance between circles represent topic similarity (approx to the original topic similarity matrix
# since we are using a two dimensional scatter plot)

# dimensionality of original (num_topics-1)^num_topics/2, multidimensional scaling does its best
# to preserve the original distance

# Extract the topic with highest security, trust and privacy

def relevent_topic(num_topics, query):
    '''
    Input:
        > num_topics: integer
        > query: term string
    '''
    print('{} topics:\n'.format(query.upper()))
    relevant = [(num, dist) for num, dist in model.show_topics(num_topics = num_topics) if len(re.findall(query, dist)) != 0]
    
    for i in relevant:
        print('Topic: {}'.format(i[0]))
        print('Distribution: {}\n'.format(i[1]))

relevent_topic(10, 'security')
relevent_topic(10, 'privacy')
relevent_topic(10, 'trust')

# Heatmap of Cos Metrics for LDA
data_lda = {i: OrderedDict(model.show_topic(i,5)) for i in range(len(model.show_topics(num_topics=10)))}

#data_lda
df_lda = pd.DataFrame(data_lda)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)

g=sns.clustermap(df_lda.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=1, figsize=(10, 12))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()

# Find the dominant topic in each sentence
# Find the topic number with the highest percentage contribution in that document

def dominant_topic(ldamodel, corpus=corpus, texts):
    # init dataframe
    topics_df = pd.DataFrame()

    # GET MAIN TOPIC IN EACH WEBPAGE
    # Get throught the pages
    for num, page in enumerate(ldamodel[corpus]):
        # Count number of list into a list
        if sum(isinstance(i, list) for i in page)>0:
            page = page[0]

        page = sorted(page, key= lambda x: (x[1]), reverse=True)
    
        for j, (topic_num, prop_topic) in enumerate(page):
            if j == 0: # => dominant topic
                # Get list prob. * keywords from the topic
                pk = ldamodel.show_topic(topic_num)
                topic_keywords = ', '.join([word for word, prop in pk])
                # Add topic number, probability, keywords and original text to the dataframe
                topics_df = topics_df.append(pd.Series([int(topic_num), np.round(prop_topic, 4),
                                                    topic_keywords, texts[num]]),
                                                    ignore_index=True)
            else:
                break
                
    # Add columns name
    topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text']

    return topics_df

target_corpus = [dictionary.doc2bow(text) for text in comment]

df_topic_keywords = dominant_topic(ldamodel=optimal_model, corpus=target_corpus, texts=df['body'])

df_topic_keywords.head(10)

# export_csv = df_topic_keywords.to_csv('output/page_topic.csv', index = None)

# Find the most representative document for each topic in order to infer the topic

df_topic_sorted = pd.DataFrame()
df_topic_grouped = df_topic_keywords.groupby('Dominant_Topic')

for i, grp in df_topic_grouped:
    # populate the sorted dataframe with the page that contributed the most to the topic
    df_topic_sorted = pd.concat([df_topic_sorted, grp.sort_values(['Perc_Contribution'], ascending = [0]).head(1)], axis = 0)
    
# Reset Index and change columns name
df_topic_sorted.reset_index(drop = True, inplace = True)
df_topic_sorted.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

df_topic_sorted.head(20)

# Topic distribution across documents
# To understand the volumne and distribution of topics in order to check how widely it was discussed

# Number of documents for each topic
topic_counts = df_topic_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for each Topic
topic_contribution = np.round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sorted[['Topic_Num', 'Keywords']].set_index(df_topic_sorted['Topic_Num'])

df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis = 1)

df_dominant_topics.reset_index(drop = True, inplace = True)
df_dominant_topics.columns = ['Topic_Num', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

df_dominant_topics.head()

# Frequency distribution of word counts in documents
# How big the documents are as a whole in term of words count

# Expand display output in terminal
pd.options.display.max_colwidth = 100

df_topic_text = dominant_topic(ldamodel=optimal_model, corpus=corpus, texts=train_words_lemmatized)

words_len = [len(words) for words in df_topic_text.Text]

plt.figure(figsize=(16,7), dpi=160)
plt.hist(words_len, bins = 28, color='navy')
plt.text(1000, 4.5, "Mean   : " + str(round(np.mean(words_len))))
plt.text(1000,  4.25, "Median : " + str(round(np.median(words_len))))
plt.text(1000,  4, "Stdev   : " + str(round(np.std(words_len))))
plt.text(1000,  3.75, "1%ile    : " + str(round(np.quantile(words_len, q=0.01))))
plt.text(1000,  3.5, "99%ile  : " + str(round(np.quantile(words_len, q=0.99))))
plt.gca().set(xlim=(0, 3000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,3000,18))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

fig, axes = plt.subplots(2,4,figsize=(15,5), dpi=160, sharex=True, sharey=True)

# for each topic
for i, ax in enumerate(axes.flatten()):
    # Extract document within the specific topic
    df_topic_sub = df_topic_text.loc[df_topic_text.Dominant_Topic == i, :]
    topic_words_len = [len(words) for words in df_topic_sub.Text]
    ax.hist(topic_words_len, bins = 5, color = cols[i])
    ax.tick_params(axis = 'y', labelcolor = cols[i], color = cols[i])
    ax.set(xlim=(0, 3000), xlabel = 'Document Word Count')
    ax.set_ylabel('Number of Documents', color = cols[i])
    ax.set_title('Topic: ' +str(i), fontdict = dict(size = 16, color = cols[i]))

fig.tight_layout()
#fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,3000,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
plt.show()

# Word coluds of Top N Keywords in Each Topic
# with the size of the words proportional to the weight

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)


fig, axes = plt.subplots(2, 4, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=22, weight = 'bold'))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

# Let's plot word counts and the weights of each keyword in the same chart
# Keep an eye on common words that occur in multiple topics and the one
# whose relative frequency is more than the weight. >> those should be added to stop_words

topics = optimal_model.show_topics(formatted=False)
data_flat = [word for page in train_words_lemmatized for word in page]

# words stored as dict keys and their count as dict values
counter = Counter(data_flat)

out = []
for num, dist in topics:
    # relative weight to the topic
    for word, weight in dist:
        out.append([word, num, weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])    

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 4, figsize=(15,5), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.20); ax.set_ylim(0, 400)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=10)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right', fontsize=7)
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=10, y=1.05)    
plt.show()

# Sentence Chart Colored by Topic
# colour each word by its representative topics
# colour document enclosing rectangle by the topic assigned

# start is included but not the end
def colour_docs(lda_model=lda_model, corpus=corpus, start=0, end=5):
    # select part of documents that your are interested in
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    # Initiate figure
    _, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)
    # Delete the the first row
    axes[0].axis('off')
    # Go throught each row of data
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1]
            # lda_model return:
            # > topic's id, and the probability that was assigned to it
            # > Most probable topics per word
            # > Phi relevance values
            topic_percs, wordid_topics, _ = lda_model[corp_cur]
            # return a tuple with word and dominant topic
            word_dominanttopic = [(lda_model.id2word[word], topic[0]) for word, topic in wordid_topics]
            # write Doc num : inside the rectangle
            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment="center",
                    # ax.transAxes fixed coordinate system (0,0) bottom left and (1.0, 1.0) top right
                    fontsize=16, color="black", transform=ax.transAxes, fontweight=700)

            # Draw Rectangle
            # find the dominant topic in for all the document
            topic_prt_sorted = sorted(topic_percs, key = lambda x: (x[1]), reverse=True)
            # (bottom, left), width, hight
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_prt_sorted[0][0]], linewidth=2))

            # Add the colored words
            word_position = 0.06  ## start after the Doc num:
            for j, (word, topics) in enumerate(word_dominanttopic):
                # write down only the first 14 words
                if  j < 5:
                    ax.text(word_position, 0.5, word, horizontalalignment='left', 
                            verticalalignment='center', fontsize=16, 
                            color=mycolors[topics], transform=ax.transAxes, fontweight=700)
                    word_position += 0.009 * len(word) # fowards word position for next iter
                    # disable the original exis
                    ax.axis('off')
            
            # Add the final three dots at the end of each row
            ax.text(word_position, 0.5, '. . .', horizontalalignment='left', 
                    verticalalignment='center', fontsize=16, color='black')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2),
                 fontsize=22, y=0.95, fontweight = 700)
    plt.tight_layout()
    plt.show()


# What are the most discussed topics in the documents
def topics_per_document(model, corpus, start=0, end=1):
    corp = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corp):
        topic_prt, _, _ = model[corp]
        dominant_topic = sorted(topic_prt, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_prt)
    return(dominant_topics, topic_percentages)


dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1) 

# Distribution of Dominant Topics in each documents
df = pd.DataFrame(dominant_topics, columns = ['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]
df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)

# Plots number of documents by dominant topic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4), dpi=120, sharey=True)

ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.tick_params(axis='x', labelsize=4)
ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
ax1.set_ylabel('Number of Documents')
ax1.set_ylim(0, 10)


# Plots topic weights for all documents
ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.tick_params(axis='x', labelsize=4)
ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

plt.show()

# t-SNE Clustering Chart
# Get topic weights
topic_weights = []
num_topics = 10  # get it from the formula
for row_list in lda_model[corpus]:
    tmp = np.zeros(num_topics)
    for i, w in row_list[0]:
        tmp[i] = w
    topic_weights.append(tmp)

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic position in each doc
topic_num = np.argmax(arr, axis=1)


# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the topic clusters using bokeh
output_notebook()
n_topics = 10
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)


# sklearn prediction >> improve the pipelines





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
