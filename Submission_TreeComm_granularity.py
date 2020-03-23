import re
import warnings
from contextlib import contextmanager
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get the passowrd
with open('password.txt', 'r') as file:
    database_password = file.readline()

@contextmanager
def mysql_connection():
    mydb = mysql.connector.connect(user="root",
                                   password=database_password,
                                   host="localhost", 
                                   port="3305",
                                   database="reddit_smarthome")

    mycursor = mydb.cursor()

    yield mycursor

    mydb.close()
    mycursor.close()


# import data
with mysql_connection() as mycursor:
    mycursor.execute('SELECT * FROM reddit_comments')
    comments = mycursor.fetchall()

with mysql_connection() as mycursor:
    mycursor.execute('SELECT * FROM reddit_submissions')
    submissions = mycursor.fetchall()


comments = pd.DataFrame(np.array(comments), 
                        columns=['id', 'link_id', 'parent_id', 'created_utc',\
                                 'body', 'author', 'permalink', 'score',\
                                 'subreddit'])

submissions = pd.DataFrame(np.array(submissions), 
                        columns=['id', 'created_utc',\
                                 'title', 'selftext', 'author', 'permalink', 'score',\
                                 'subreddit', 'num_comments', 'link_id'])

# Randomly select num_comments for each subreddit where link_id == parent_id 
# Fist tier comments
tmp = comments[comments['link_id'] == comments['parent_id']]
df = pd.concat([tmp[tmp['subreddit'] == 'smarthome'].sample(n=5000, random_state=123),
                            tmp[tmp['subreddit'] == 'homeautomation'].sample(n=5000, random_state=123)])

# add a parent id column without t#_
comments['id_parent_copy'] = [re.sub('.+_', '', x) for x in comments['parent_id']]
# initiate an empty database 
df_tree = pd.DataFrame(columns = ['tree_ids', 'tree_bodies'])

try:
    for first_com in df.loc[:, 'link_id'].unique(): # Comments from 7283 submission
        # For each parent_id get all comments with the same link_id
        # Get all comments within the same submission
        first_tier = comments[comments['link_id'] == first_com]
        # Initiate an empty tree
        tree_ids = []
        tree_bodies = []
        # Isolete the selected first tier comment >> could be more than one
        for init_i in list(df['id'][df['link_id'] == first_com]):
            tree_ids.append(init_i)
            tree_bodies.append(''.join(list(df['body'][df['id'] == init_i])))

            # concatenate all the children 
            i = []
            i.append(init_i)
            while not comments[comments['id_parent_copy'].isin(i)].empty:
                # all rows in sorted tmp inserted in tree_ids and tree bodies
                # all comments in the same tier are concatenate to each other
                sorted_tmp = comments[comments['id_parent_copy'].isin(i)].sort_values(by = ['created_utc'], ascending=False)
                num = list(df['id'][df['link_id'] == first_com]).index(init_i)
                tree_ids[num] += ' <NEW TIER> ' + ' <SAME TIER> '.join(list(sorted_tmp['link_id']))
                tree_bodies[num] += ' <NEW TIER> ' + ' <SAME TIER> '.join([str(elm) for elm in list(sorted_tmp['body'])])
                i = list(sorted_tmp['id'])

        # store in a new database
        for n_row in range(len(tree_ids)):
            df_tree = df_tree.append({'tree_ids': tree_ids[n_row], 'tree_bodies': tree_bodies[n_row]}, ignore_index=True)
except:
    first_com



# ADD SUBMISSION INFO
# extract first id for all rows
df_tree['id'] = [re.sub('\\s.*', '',x) for x in  df_tree['tree_ids']]

df_tree = df_tree.merge(comments.loc[:, ['id','link_id']], on = 'id')
df_tree = df_tree.merge(submissions.loc[:, ['link_id', 'title', 'selftext']], on='link_id')

df_tree['text'] =[' <SUB> '.join([df_tree['title'][num], df_tree['selftext'][num], df_tree['tree_bodies'][num]]) for num in range(df_tree.shape[0])]


df_tree.to_csv('df_tree.csv', index=False)