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
import joblib
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel, LdaModel
from gensim.test.utils import datapath
from gensim.matutils import corpus2csc
import NLP_visualization as NLP_vis
from gensim.matutils import corpus2csc, Sparse2Corpus

## lOAD
# train data
df_train = pd.read_csv('.\\Datasource_backup\\sub_onetree_train.csv')

# trained bigram
filename = datapath('train_bigram\\nb5_na04_bigram.pkl')
with open(filename, "rb") as f:
    train_bigram = pickle.load(f)

# trained dictionary
dictionary = corpora.Dictionary.load(datapath("vocabulary\\nb5_na04"))
NLP_vis.vocabulary_freq_words(dictionary, False, 30)

# # trained models
# filename = datapath('train_models\\nb5_na04_models.joblib')
# train_models = joblib.load(filename)


# # transform documents in a document-term matrix
# X_train = train_bigram.transform(df_train['clean_text'].tolist())
# corpus = Sparse2Corpus(X_train, documents_columns=False)

# # Words used in how many texts?
# NLP_vis.vocabulary_descriptive(dictionary, corpus)


# from os import listdir
# model_name = [x for x in listdir(datapath('train_models\\nb5_na04\\')) 
#             if not x.endswith(tuple(["npy", "id2word", "state"]))]

# models = {}
# for name in model_name:
#     models[name] = LdaModel.load(datapath('train_models\\nb5_na04\\{}'.format(name)))


# # models_dict = {}
# # for i in os.listdir(datapath('LdaMallet_model')):
# #     models_dict[i] = LdaMallet.load(datapath('LdaMallet_model\\{}'.format(i)))

# # # Calculate evaluation metrics
# # new_dict = {}
# # for i in models_dict.keys():
# #     key = int(re.sub('[^\\d]', '', i))
# #     new_dict[key] = models_dict[i]

# model_1_001 = {}
# for i in range(5, 501, 5):
#     model_1_001[i] = LdaModel.load(datapath('model_1_001\\{:d}'.format(i)))

# model_10_01 = {}
# for i in range(5, 501, 5):
#     model_10_01[i] = LdaModel.load(datapath('model_10_01\\{:d}'.format(i)))

# model_50_05 = {}
# for i in range(5, 501, 5):
#     model_50_05[i] = LdaModel.load(datapath('model_50_05\\{:d}'.format(i)))

# model_01_0001 = {}
# for i in range(5, 501, 5):
#     model_01_0001[i] = LdaModel.load(datapath('model_01_0001\\{:d}'.format(i)))

# coherence_gensim_c_v = []
# cao_juan_2009 =[]
# arun_2010 =[]
# coherence_mimno_2011 = []

# # for i in sorted(new_dict.keys()):
# #     coherencemodel = CoherenceModel(model = new_dict[i], texts = comments,
# #                                     dictionary = dictionary, coherence = 'c_v')
# #     coherence_gensim_c_v.append(coherencemodel.get_coherence())

# #     cao_juan_2009.append(evaluate.metric_cao_juan_2009(new_dict[i].get_topics()))

# #     arun_2010.append(evaluate.metric_arun_2010(new_dict[i].get_topics(),  
# #                         np.array([x.transpose()[1] for x in np.array(list(new_dict[i].load_document_topics()))]),
# #                         np.array([len(x) for x in comments])))

# #     coherence_mimno_2011.append(evaluate.metric_coherence_mimno_2011(new_dict[i].get_topics(), 
# #                                                                 corpus2csc(corpus).transpose(), return_mean=True))
# num=0
# for model_dict in [model_1_001, model_10_01, model_50_05]:
#     for i in sorted(model_dict.keys()):
#         coherencemodel = CoherenceModel(model = model_dict[i], texts = comments,
#                                         dictionary = dictionary, coherence = 'c_v')
#         coherence_gensim_c_v.append(coherencemodel.get_coherence())

#         cao_juan_2009.append(evaluate.metric_cao_juan_2009(model_dict[i].get_topics()))

#         arun_2010.append(evaluate.metric_arun_2010(model_dict[i].get_topics(),  
#                             np.array([x.transpose()[1] for x in np.array(list(model_dict[i].get_document_topics(corpus, minimum_probability=0)))]),
#                             np.array([len(x) for x in comments])))

#         coherence_mimno_2011.append(evaluate.metric_coherence_mimno_2011(model_dict[i].get_topics(), 
#                                                                     corpus2csc(corpus).transpose(), return_mean=True))

#     if num == 0:
#         print('Saving evaluation metrics for alpha=1/k and beta = 0.01')
#         coherence_gensim_c_v_1_001 = coherence_gensim_c_v
#         cao_juan_2009_1_001 = cao_juan_2009
#         arun_2010_1_001 = arun_2010
#         coherence_mimno_2011_1_001 = coherence_mimno_2011
    
#     if num == 1:
#         print('Saving evaluation metrics for alpha=10/k and beta = 0.1')
#         coherence_gensim_c_v_10_01 = coherence_gensim_c_v
#         cao_juan_2009_10_01 = cao_juan_2009
#         arun_2010_10_01 = arun_2010
#         coherence_mimno_2011_10_01 = coherence_mimno_2011

#     if num == 2:
#         print('Saving evaluation metrics for alpha=50/k and beta = 0.5')
#         coherence_gensim_c_v_50_05 = coherence_gensim_c_v
#         cao_juan_2009_50_05 = cao_juan_2009
#         arun_2010_50_05 = arun_2010
#         coherence_mimno_2011_50_05 = coherence_mimno_2011

#     coherence_gensim_c_v = []
#     cao_juan_2009 =[]
#     arun_2010 =[]
#     coherence_mimno_2011 = []
#     num+=1

# # Save evaluation metrics
# pd.DataFrame(coherence_gensim_c_v_1_001).to_csv('Evaluation\\model_1_001\\coherence_gensim_c_v_1_001.csv', header=None, index=False)
# pd.DataFrame(cao_juan_2009_1_001).to_csv('Evaluation\\model_1_001\\cao_juan_2009_1_001.csv', header=None, index=False)
# pd.DataFrame(arun_2010_1_001).to_csv('Evaluation\\model_1_001\\arun_2010_1_001.csv', header=None, index=False)
# pd.DataFrame(coherence_mimno_2011_1_001).to_csv('Evaluation\\model_1_001\\coherence_mimno_2011_1_001.csv', header=None, index=False)

# pd.DataFrame(coherence_gensim_c_v_10_01).to_csv('Evaluation\\model_10_01\\coherence_gensim_c_v_10_01.csv', header=None, index=False)
# pd.DataFrame(cao_juan_2009_10_01).to_csv('Evaluation\\model_10_01\\cao_juan_2009_10_01.csv', header=None, index=False)
# pd.DataFrame(arun_2010_10_01).to_csv('Evaluation\\model_10_01\\arun_2010_10_01.csv', header=None, index=False)
# pd.DataFrame(coherence_mimno_2011_10_01).to_csv('Evaluation\\model_10_01\\coherence_mimno_2011_10_01.csv', header=None, index=False)

# pd.DataFrame(coherence_gensim_c_v_50_05).to_csv('Evaluation\\model_50_05\\coherence_gensim_c_v_50_05.csv', header=None, index=False)
# pd.DataFrame(cao_juan_2009_50_05).to_csv('Evaluation\\model_50_05\\cao_juan_2009_50_05.csv', header=None, index=False)
# pd.DataFrame(arun_2010_50_05).to_csv('Evaluation\\model_50_05\\arun_2010_50_05.csv', header=None, index=False)
# pd.DataFrame(coherence_mimno_2011_50_05).to_csv('Evaluation\\model_50_05\\coherence_mimno_2011_50_05.csv', header=None, index=False)

# for i in sorted(model_01_0001.keys()):
#     coherencemodel = CoherenceModel(model = model_01_0001[i], texts = comments,
#                                     dictionary = dictionary, coherence = 'c_v')
#     coherence_gensim_c_v.append(coherencemodel.get_coherence())

#     cao_juan_2009.append(evaluate.metric_cao_juan_2009(model_01_0001[i].get_topics()))

#     arun_2010.append(evaluate.metric_arun_2010(model_01_0001[i].get_topics(),  
#                         np.array([x.transpose()[1] for x in np.array(list(model_01_0001[i].get_document_topics(corpus, minimum_probability=0)))]),
#                         np.array([len(x) for x in comments])))

#     coherence_mimno_2011.append(evaluate.metric_coherence_mimno_2011(model_01_0001[i].get_topics(), 
#                                                                 corpus2csc(corpus).transpose(), return_mean=True))

# pd.DataFrame(coherence_gensim_c_v).to_csv('Evaluation\\model_01_0001\\coherence_gensim_c_v_01_0001.csv', header=None, index=False)
# pd.DataFrame(cao_juan_2009).to_csv('Evaluation\\model_01_0001\\cao_juan_2009_01_0001.csv', header=None, index=False)
# pd.DataFrame(arun_2010).to_csv('Evaluation\\model_01_0001\\arun_2010_01_0001.csv', header=None, index=False)
# pd.DataFrame(coherence_mimno_2011).to_csv('Evaluation\\model_01_0001\\coherence_mimno_2011_01_0001.csv', header=None, index=False)

# # LOAD DATA

# coherence_gensim_c_v = pd.read_csv('Evaluation\\50_0.5\\coherence_gensim_c_v.csv', header=None)[0].values.tolist()
# cao_juan_2009 = pd.read_csv('Evaluation\\50_0.5\\(cao_juan_2009.csv', header=None)[0].values.tolist()
# arun_2010 = pd.read_csv('Evaluation\\50_0.5\\arun_2010.csv', header=None)[0].values.tolist()
# coherence_mimno_2011 = pd.read_csv('Evaluation\\50_0.5\\coherence_mimno_2011.csv', header=None)[0].values.tolist()

# #Plot 
# def evaluation_plot(arun, cao_juan, c_v, mimno, alpha, beta):
#     foo = np.arange(5, 200, 5)
#     print("Arun 2010:'\t' min {}'\t' @topic {}".format(min(arun), foo[np.argmin(arun)]))
#     print("Cao Juan 2009:'\t' min {}'\t' @topic {}".format(min(cao_juan), foo[np.argmin(cao_juan)]))
#     print("Choerence Gensim c_v:'\t' max {}'\t' @topic {}".format(max(c_v), foo[np.argmax(c_v)]))
#     print("Choerence Mimno 2011:'\t' max {}'\t' @topic {}".format(max(mimno), foo[np.argmax(mimno)]))
#     fig, axs = plt.subplots(4, 1, sharex=True)
#     axs[0].plot(range(5, 200, 5), arun)
#     axs[0].set_title('MINIMIZE: arun_2010')
#     axs[1].plot(range(5, 200, 5), cao_juan)
#     axs[1].set_title('MINIMIZE: cao_juan_2009')
#     axs[2].plot(range(5, 200, 5), c_v)
#     axs[2].set_title('MAXIMIZE: coherence_gensim_c_v')
#     axs[3].plot(range(5, 200, 5), mimno)
#     axs[3].set_title('MAXIMIZE: coherence_mimno_2011')
#     axs[3].set_xlabel('num. topics (k)')
#     axs[3].set_xlim(0, 200)
#     axs[3].set_xticks(np.arange(0, 200, 5))
#     axs[3].set_xticklabels(labels=np.arange(0, 200, 5), rotation=45)
#     fig.suptitle('Evaluation results for alpha={}/k, beta={}'.format(alpha, beta))
#     plt.show()

# evaluation_plot(arun_2010, cao_juan_2009, coherence_gensim_c_v, coherence_mimno_2011, 50, 0.5)