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

comments = pd.DataFrame(np.array(comments), 
                        columns=['id', 'link_id', 'parent_id', 'created_utc',\
                                 'body', 'author', 'permalink', 'score',\
                                 'subreddit', 'category'])

# Randomly select num_comments for each subreddit where link_id == parent_id 
# Fist tier comments
tmp = comments[comments['link_id'] == comments['parent_id']]
df = pd.concat([tmp[tmp['subreddit'] == 'smarthome'].sample(n=5000, random_state=123),
                            tmp[tmp['subreddit'] == 'homeautomation'].sample(n=5000, random_state=123)])

# add a parent id column without t#_
comments['id_parent_copy'] = [re.sub('.+_', '', x) for x in comments['parent_id']]
# initiate an empty database 
df_tree = pd.DataFrame(columns = ['tree_ids', 'tree_bodies'])

for first_com in df.loc[:, 'link_id'].unique()[:3]: # Comments from 7283 submission
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
            tree_bodies[num] += ' <NEW TIER> ' + ' <SAME TIER> '.join(list(sorted_tmp['body']))
            i = list(sorted_tmp['id'])

    # store in a new database
    for n_row in range(len(tree_ids)):
        df_tree = df_tree.append({'tree_ids': tree_ids[n_row], 'tree_bodies': tree_bodies[n_row]}, ignore_index=True)


# extract all comments with the same parent_id


# Cleaning up the comments
# nltk.download('stopwords')  # (run python console)
# nltk.download('wordnet')  # (run python console)
# python3 -m spacy download en  # (run in terminal)

# Extracting URLs: external links often informative, but they add unwanted noise to our NLP model
# Strip out hyperlinks and copy thme in a new column URL

# remove rows with less than two words
df = df[df['body'].map(lambda x: len(str(x).strip().split())) >= 2]

# Find URL
def find_URL(comment):
    return re.findall(r'((?:https?:\/\/)(?:\w[^\)\]\s]*))', comment)

df['URL'] = df.body.apply(find_URL)

# create a colummn with pre-processed:
df['clean_body'] = [re.sub(r"((?:https?:\/\/)(?:\w[^\)\]\s]*))",'', x) for x in df['body']]

#NOTE: consider internal hyphen as full words. "Technical vocabulary"

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['etc', 'however', 'there', 'also', 'digit'])

#We specify the stemmer or lemmatizer we want to use
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem

# Remove comments where 70% words are not part of the english vocabulary
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

def clean_comment(comment, lemma=True, del_tags = ['NUM', 'PRON']):
    comment = comment.lower() # ? consider to make general the name of companies or decives
    comment = re.sub(r'&gt', ' ', comment) # remove all copied text into a comment '&gt'
    comment = re.sub(r'&amp;', ' ', comment) # & charcter
    comment = re.sub(r'#x200b;', ' ', comment) # zero-witdh space
    comment = re.sub(r'remindme![\w\s\W]*$', ' ', comment) # remove call to remind me bot
    comment = re.sub(r'[^\s\w\$]', ' ', comment) # strip out everything (punctuation) that is not Unicode whitespace or word character
    comment = re.sub(r'[_]', ' ', comment) # remove underscores around a words (italics)
    comment = re.sub(r'[0-9]+', 'digit', comment) # remove digits
    comment = re.sub(r'\$', 'dollar', comment)

    # detect no english comments and remove them (14714)
    #nltk.download('words')
    text_vocab = set(w for w in comment.strip().split() if w.isalpha())
    unusual = text_vocab.difference(english_vocab) 

    # empty comments where 70% words not english, slangs, deleted
    try:
        if len(unusual)/len(text_vocab) > 0.7:
            comment = ''
    except ZeroDivisionError:
        pass

    # remove stop_words
    comment_token_list = [word for word in comment.strip().split() if word not in stop_words]
    
    # keeps word meaning: important to infer what the topic is about
    if lemma == True:
        # Initialize spacy 'en' model
        nlp = spacy.load('en_core_web_sm')
        # https://spacy.io/api/annotation
        comment_text = nlp(' '.join(comment_token_list))
        comment_token_list = [token.lemma_ for token in comment_text if token.pos_ not in del_tags]
    
    # harsh to the root of the word
    else:
        comment_token_list = [word_rooter(word) for word in comment_token_list]

    comment = ' '.join(comment_token_list)
    
    return comment

# Apply function to clean the comment
df['clean_body'] = df.clean_body.apply(clean_comment)

# NOTE: missplellings and slangs
# NOTE: the length of your individual documents should not be too inbalanced and not too short for Topic Modeling
# remove rows with less than 2 word
df = df[df['clean_body'].map(lambda x: len(str(x).strip().split())) >= 2]

# delete RemindMeBot  
df = df[df['author'] != 'RemindMeBot']

#df.to_csv('50miclean.csv', index = True)
# Train Bigram and Trigram Models
# higher threshold fewer phrases.
comment_token_list = df['clean_body'].map(lambda x: str(re.sub(r'[_]', ' ',x)).strip().split())

bigram = gensim.models.Phrases(comment_token_list, min_count=200, threshold=25)
trigram = gensim.models.Phrases(bigram[comment_token_list], min_count=200, threshold=25)

dist = nltk.FreqDist(
    [word for comment in trigram[bigram[comment_token_list]] for word in comment if '_' in word])

# Sort frequency
print('Sorted trigrams: \n')
print(sorted(dist.items(), key=lambda x: x[1], reverse=True))
print('-'*20)

df['clean_body'] = list(trigram[bigram[comment_token_list]])
df['clean_body'] = df['clean_body'].map(lambda x: ' '.join(x))

# write df
# df.to_csv('trigram_comments.csv', index=True)

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

#freq_words(df['clean_body'], False, 50)

# Frequency distribution of word counts in documents
df = pd.read_csv('preprocessed_comments10k.csv', index_col=0)

words_len = [len(words) for words in df.clean_body]

plt.figure(figsize=(16,7), dpi=160)
plt.hist(words_len, bins = 500, color='navy')
plt.text(1000, 450, "Mean   : " + str(round(np.mean(words_len))))
plt.text(1000, 400, "Median : " + str(round(np.median(words_len))))
plt.text(1000, 350, "Stdev   : " + str(round(np.std(words_len))))
plt.text(1000, 300, "1%ile    : " + str(round(np.quantile(words_len, q=0.01))))
plt.text(1000,  250, "99%ile  : " + str(round(np.quantile(words_len, q=0.99))))
plt.gca().set(xlim=(0, 4100), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,max(words_len),18))
plt.title('Distribution of Document Word Counts - 10k', fontdict=dict(size=22))
plt.savefig('WordCount')
plt.show()

# write the csv file
df.to_csv('preprocessed_comments100mi.csv', index = True)
