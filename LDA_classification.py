import os
import re
import pickle
import pandas as pd
import numpy as np
import warnings
from scipy.spatial import distance
from gensim.test.utils import datapath
import gensim.corpora as corpora
from gensim.matutils import Sparse2Corpus

warnings.filterwarnings("ignore", category=FutureWarning)

# Load selected model
models_dir = datapath("train_models\\")
selected_model = "nb5_na04_a1_b1_models.pkl"
with open(os.path.join(models_dir, selected_model), "rb") as handle:
    model = pickle.load(handle)

a1_b1_k20 = model["a1_b1_k20"]

# Load dictionary 
dictionary = corpora.Dictionary.load(datapath("vocabulary\\{}".format("nb5_na04")))

# load trained bigram
filename = datapath('train_bigram\\{}_bigram.pkl'.format("nb5_na04"))
with open(filename, "rb") as f:
    train_bigram = pickle.load(f)
    
# Load train data
df_train = pd.read_csv('.\\Datasource_backup\\sub_onetree_train.csv')

X_train = train_bigram.transform(df_train['clean_text'].tolist())
corpus = Sparse2Corpus(X_train, documents_columns=False)

# Calculate Jensen-Shannon distance between two probability distributions using scipy.stats.entropy.
# Create Document - Topic Matrix

# column names
topicnames = ["Topic" + str(i) for i in range(len(a1_b1_k20.print_topics(num_topics=20)))]

# index names
docnames = ["Doc" + str(i) for i in range(df_train.shape[0])]

# Make the pandas dataframe
doc_topic = pd.DataFrame(columns=topicnames, index=docnames)

# Populate the matrix with topic probability distribution for a document
for doc_num, dist in enumerate(a1_b1_k20[corpus]):
    for i in dist[0]:
        doc_topic.iloc[doc_num,i[0]] = i[1]

doc_topic = doc_topic.fillna(0)

# DOCUMENT BY DOCUMENT JENSENSHANNON CALCULATION
# dictionary implementation
js_dict = {}

for r in np.array(range(df_train.shape[0])):
    for c in np.array(range(df_train.shape[0])):
        if c <= r:
            pass
        else:
            js_dict[str(r)+'-'+str(c)] = distance.jensenshannon(doc_topic.iloc[r,:], doc_topic.iloc[c,:])

jsd_file = datapath("inspection\\nb5_na04_JSD_dict.pkl")
with open(jsd_file, "wb") as handle:
    pickle.dump(js_dict, handle)

# Find the topic with the highest contribution for each document
def dominant_topic(ldamodel, corpus, document):
    # init dataframe
    topics_df = pd.DataFrame()

    # GET MAIN TOPIC IN EACH DOCUMENT
    # Get throught the pages
    for num, doc in enumerate(ldamodel[corpus]):
        # Count number of list into a list
        if sum(isinstance(i, list) for i in doc)>0:
            doc = doc[0]

        doc = sorted(doc, key= lambda x: (x[1]), reverse=True)
    
        for j, (topic_num, prop_topic) in enumerate(doc):
            if j == 0: # => dominant topic
                # Get list prob. * keywords from the topic
                pk = ldamodel.show_topic(topic_num)
                topic_keywords = ', '.join([word for word, prop in pk])
                # Add topic number, probability, keywords and original text to the dataframe
                topics_df = topics_df.append(pd.Series([int(topic_num), np.round(prop_topic, 4),
                                                    topic_keywords, document[num]]),
                                                    ignore_index=True)
            else:
                break
                
    # Add columns name
    topics_df.columns = ['Dominant_Topic', '%_Contribution', 'Topic_Keywords', 'Text']

    return topics_df


df_dominant_topic = dominant_topic(a1_b1_k20, corpus, df_train['text'])

# Find the most representative document for each topic
df_topic_sorted = pd.DataFrame()
df_topic_grouped = df_dominant_topic.groupby('Dominant_Topic')

for i, grp in df_topic_grouped:
    # populate the sorted dataframe with the document that contributed the most to the topic
    df_topic_sorted = pd.concat([df_topic_sorted, grp.sort_values(['%_Contribution'], ascending = [0]).head(1)], axis = 0)
    
# Reset Index and change columns name
df_topic_sorted.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

df_topic_sorted.loc[df_topic_sorted["Topic_Num"].isin([19, 18, 9, 7, 4, 16, 15, 11, 5])]

# save documents for each topic with JSD <= 0.4 compared to the most representative
# documents for each topics

doc_topic = df_topic_sorted.loc[df_topic_sorted["Topic_Num"].isin([19, 18, 9, 7, 
                                                                    4, 16, 15, 11, 5]),"Topic_Num"]

for n in range(doc_topic.shape[0]):
    idx = doc_topic.index[n]
    topic = int(doc_topic.iloc[n])

    # Get all JSD calculations for documents related to the reference document
    # with JSD <= 0.4
    topic_docs = {k: v for k, v in js_dict.items() if str(idx) in re.findall(r'(\d*)-(\d*)', k)[0]}
    topic_docs = {k: v for k, v in topic_docs.items() if v <= 0.4}

    k_1 = []; k_2 = []
    for k in list(topic_docs.keys()):
        k_1.append(int(re.findall(r'(\d*)-(\d*)', k)[0][0]))
        k_2.append(int(re.findall(r'(\d*)-(\d*)', k)[0][1]))

    topic_df = df_train.iloc[list(set(k_1).union(set(k_2))), :]

    JSD_col = {int(re.sub(r'-*({})-*'.format(str(idx)), '', k)): v for k, v in topic_docs.items()}

    # Add the reference document
    JSD_col[idx] = 0
    JSD_col = pd.DataFrame.from_dict(JSD_col, orient='index', columns=["JSD"])

    topic_df = pd.merge(topic_df, JSD_col, left_index=True, right_index=True)

    topic_df.to_csv(datapath("inspection\\nb5_na04_topic_{}_df.csv".format(str(topic))), index=True)