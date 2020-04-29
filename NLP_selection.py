# LOAD EVALUATION DATA
import pandas as pd
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gensim.test.utils import datapath

eval_dir = datapath("evaluation\\")
filename = [f for f in os.listdir(eval_dir) if f.startswith("{}".format("nb5_na04"))]

# Initiate an empty dataframe
evaluation = pd.DataFrame()
for file in filename:
    new = pd.read_csv(eval_dir + file)
    # add a new column wirh alpha and eta parameters
    new['alpha_eta'] = re.findall(".*_", file)[0][9:-1]
    evaluation = pd.concat([evaluation, new])

#Plot evaluation metrics
# https://matplotlib.org/3.2.1/tutorials/intermediate/gridspec.html
def evaluation_plot(eval_dataframe):
    fig = plt.figure(figsize=(20, 20))

    combinations = len(eval_dataframe.alpha_eta.unique())
    # Compute the number of rows and columns to display on the grid
    display_rows = math.floor(np.sqrt(combinations))
    display_cols = math.ceil(combinations/display_rows)
    outer = fig.add_gridspec(display_rows, display_cols, wspace=0.2, hspace=0.2)
    
    # By combinations of alpha and eta
    for i in range(combinations):
        inner = outer[i].subgridspec(4, 1, wspace=0.1, hspace=0.6)
        ax0 = fig.add_subplot(outer[i])
        outer_text = eval_dataframe.alpha_eta.unique()[i]
        ax0.set_title("{} \n".format(outer_text), fontsize = 10, color = "black", fontweight = "bold")
        ax0.axis('off')


        # subset data based on the specific combination of alpha and eta
        tmp_comb = eval_dataframe.where(eval_dataframe.alpha_eta == eval_dataframe.alpha_eta.unique()[i]).dropna()
        # By evluation metrics
        for j in range(4):
            ax = fig.add_subplot(inner[j])
            ax.plot(tmp_comb.iloc[:, 0], tmp_comb.iloc[:, j+1])
            text = tmp_comb.columns[j+1]
            if text == "coherence_gensim_c_v":
                text = text + ": MAX {}".format(np.round(max(tmp_comb.coherence_gensim_c_v),3)) + \
                        " @TOPIC {}".format(int(tmp_comb.iloc[np.argmax(np.array(tmp_comb.coherence_gensim_c_v)), 0]))
            elif text == 'cao_juan_2009':
                text = text + ": MIN {}".format(np.round(min(tmp_comb.cao_juan_2009),3)) + \
                        " @TOPIC {}".format(int(tmp_comb.iloc[np.argmin(np.array(tmp_comb.cao_juan_2009)), 0]))
            elif text == 'arun_2010':
                text = text + ": MIN {}".format(np.round(min(tmp_comb.arun_2010),3)) + \
                        " @TOPIC {}".format(int(tmp_comb.iloc[np.argmin(np.array(tmp_comb.arun_2010)), 0]))
            elif text == 'coherence_mimno_2011':
                text = text + ": MAX {}".format(np.round(max(tmp_comb.coherence_mimno_2011),3)) + \
                        " @TOPIC {}".format(int(tmp_comb.iloc[np.argmax(np.array(tmp_comb.coherence_mimno_2011)), 0]))
            else:
                text = 'ERROR'

            ax.set_title(text, fontsize = 8, color = "red", fontweight = "bold")
            # ax.set_xticks([])
            # ax.set_yticks([])
            fig.add_subplot(ax)

    name = re.sub("_.*", "", eval_dataframe.alpha_eta.unique()[0])
    plt.savefig(os.getcwd() + "\\Figure\\evaluation_{}.png".format(name))

# Visualize plot
evaluation_a001 = evaluation.where(evaluation.alpha_eta.str.startswith("a001_")).dropna()
evaluation_plot(evaluation_a001)

evaluation_a01 = evaluation.where(evaluation.alpha_eta.str.startswith("a01_")).dropna()
evaluation_plot(evaluation_a01)

evaluation_a1 = evaluation.where(evaluation.alpha_eta.str.startswith("a1_")).dropna()
evaluation_plot(evaluation_a1)

evaluation_a10 = evaluation.where(evaluation.alpha_eta.str.startswith("a10_")).dropna()
evaluation_plot(evaluation_a10)
