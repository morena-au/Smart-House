# TENSORFLOW SETUP
# conda create -p G:\SmartHome\tf tensorflow=1.15 python=3.6
# conda install tensorflow-hub
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# Get the data
inspection_path = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\inspection"
topic_19_df = pd.read_csv(inspection_path + "\\nb5_na04_topic_19_df.csv")

#change column name
new_columns = topic_19_df.columns.values
new_columns[0] = 'raw_index'
topic_19_df.columns = new_columns

# ELMo can receive a list of sentence strings or a list of lists 
doc_list = [x for x in topic_19_df["clean_text"]]

# Create sentence embeddings
url = "https://tfhub.dev/google/elmo/3"
embed = hub.Module(url)

# run through the document list and return the default 
# output (1024 dimension document vectors).
def elmo_vectors(x):
  embeddings = embed(x, signature="default", as_dict=True)["default"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(embeddings)

# split the list of documents into batches of 3 samples each.
# to avoid running out of computational resources (memory) 

# pass these batches sequentially to 
# the function elmo_vectors( ).
doc_batch = [doc_list[i:i+3] for i in range(0,len(doc_list),3)]

# Extract ELMo embeddings
elmo_train = [elmo_vectors(x) for x in doc_batch]

# we can concatenate all the vectors back to a single array
elmo_train_new = np.concatenate(elmo_train, axis = 0)

# save the ELMo vectors
ELMo_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\ELMo_trained.pkl"

with open(ELMo_file, "wb") as handle:
    pickle.dump(elmo_train_new, handle)

# Create a semantic search engine
# search through the text not by keywords but by semantic closeness to our search query.
# > First we take a search query and run ELMo over it
trust_query = elmo_vectors(['trust'])

# save the ELMo vectors
trust01_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\trust_01.pkl"

with open(trust01_file, "wb") as handle:
    pickle.dump(trust_query, handle)


# > First we take a search query and run ELMo over it
trust_query = elmo_vectors(['trust in smart device'])

# save the ELMo vectors
trust02_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\trust_02.pkl"

with open(trust02_file, "wb") as handle:
    pickle.dump(trust_query, handle)


trust_query = elmo_vectors(['trust into companies offering smart home products and services'])

# save the ELMo vectors
trust03_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\trust_03.pkl"

with open(trust03_file, "wb") as handle:
    pickle.dump(trust_query, handle)

## SECURITY
security_query = elmo_vectors(['concerns with regard to data security'])

# save the ELMo vectors
security01_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\security_01.pkl"

with open(security01_file, "wb") as handle:
    pickle.dump(security_query, handle)

## PRIVACY
privacy_query = elmo_vectors(['privacy concerns'])

# save the ELMo vectors
privacy01_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\privacy_01.pkl"

with open(privacy01_file, "wb") as handle:
    pickle.dump(privacy_query, handle)
