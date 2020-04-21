import re
import warnings
import os
from tmtoolkit.topicmod import tm_lda, evaluate
import zstandard as zstd
import pickle
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import argv
from gensim.models import CoherenceModel, LdaModel
from gensim.test.utils import datapath
from gensim.matutils import corpus2csc
import NLP_visualization as NLP_vis
from gensim.matutils import corpus2csc, Sparse2Corpus

# avaid creating subprocesses recursively
if __name__ == '__main__':
    # unpacks argv
    _, vocabulary, alpha = argv 

    ## lOAD
    # train data
    df_train = pd.read_csv('.\\Datasource_backup\\sub_onetree_train.csv')
    print("Load train data... \n")

    # trained bigram
    filename = datapath('train_bigram\\{}_bigram.pkl'.format(vocabulary))
    with open(filename, "rb") as f:
        print("Load train bigram... \n")
        train_bigram = pickle.load(f)

    # trained dictionary
    dictionary = corpora.Dictionary.load(datapath("vocabulary\\{}".format(vocabulary)))
    print("Load dictionary... \n")
    #NLP_vis.vocabulary_freq_words(dictionary, False, 30)

    # trained models
    tmp_dir = datapath('train_models\\')
    filename = [f for f in os.listdir(tmp_dir) if f.startswith("{}".format(vocabulary))]

    # MemoryError: suhset based on alpha parameter
    filename = [f for f in os.listdir(tmp_dir) if f.startswith("{}_{}".format(vocabulary, alpha))]
 
    for file in filename:
        with open(os.path.join(tmp_dir, file), "rb") as handle:
            print("Load train models {}... \n".format(file))
            train_models = pickle.load(handle)
            print("loaded train models {} of 40 \n".format(len(train_models)))

        # transform documents in a document-term matrix
        X_train = train_bigram.transform(df_train['clean_text'].tolist())
        corpus = Sparse2Corpus(X_train, documents_columns=False)

        # Words used in how many texts?
        #NLP_vis.vocabulary_descriptive(dictionary, corpus)

        def corpus2token_text(corpus, dictionary):
            nested_doc = []
            texts = []
            for doc in corpus:
                nested_doc.append([[dictionary[k]]*v for (k, v) in doc])
            for doc in nested_doc:
                texts.append([item for sublist in doc for item in sublist])
            return texts

        texts = corpus2token_text(corpus, dictionary)

        # Calculate evaluation metrics

        coherence_gensim_c_v = []
        cao_juan_2009 =[]
        arun_2010 =[]
        coherence_mimno_2011 = []

        for i in train_models.keys():
            print("Calculate Coherence gensim cv in {}...".format(i))
            coherencemodel = CoherenceModel(model = train_models[i], texts = texts,
                                            dictionary = dictionary, coherence = 'c_v')
            coherence_gensim_c_v.append(coherencemodel.get_coherence())
            

            print("Calculate Cao Juan 2009...")
            cao_juan_2009.append(evaluate.metric_cao_juan_2009(train_models[i].get_topics()))

            print("Calculate Arun 2010...")
            arun_2010.append(evaluate.metric_arun_2010(train_models[i].get_topics(),  
                                np.array([x.transpose()[1] for x in np.array(list(train_models[i].get_document_topics(corpus, minimum_probability=0)))]),
                                np.array([len(x) for x in texts])))

            print("Calculate Coherence Mimno 2011...  \n")
            coherence_mimno_2011.append(evaluate.metric_coherence_mimno_2011(train_models[i].get_topics(), 
                                                                        corpus2csc(corpus).transpose(), return_mean=True))

        # Save evaluation metrics
        evaluation_metrics = pd.DataFrame({
            'num_topics': list(range(5, 201, 5)),
            'coherence_gensim_c_v': coherence_gensim_c_v,
            'cao_juan_2009': cao_juan_2009, 
            'arun_2010': arun_2010, 
            'coherence_mimno_2011': coherence_mimno_2011})

        eval_dir = datapath('evaluation\\')
        print("save evaluation metrics in...  \n{}".format(eval_dir))
        evaluation_metrics.to_csv(eval_dir + re.findall(".*_", file)[0][:-1] + "_eval.csv", index = False)