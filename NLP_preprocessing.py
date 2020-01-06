import re
import warnings
from contextlib import contextmanager

import mysql.connector
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

# Randomly select 5000 comments for each subreddit
df = pd.concat([comments[comments['subreddit'] == 'smarthome'].sample(n=5000, random_state=123),
                            comments[comments['subreddit'] == 'homeautomation'].sample(n=5000, random_state=123)])

# Cleaning up the comments
# nltk.download('stopwords')  # (run python console)
# nltk.download('wordnet')  # (run python console)
# python3 -m spacy download en  # (run in terminal)

# Extracting URLs: external links often informative, but they add unwanted noise to our NLP model
# Strip out hyperlinks and copy thme in a new column URL

# Find URL
def find_URL(comment):
    return re.findall(r'((?:https?:\/\/)(?:\w[^\)\]\s]*))', comment)

# apply the function on the body column
df['URL'] = df.body.apply(find_URL)

# create a colummn with pre-processed try:
df['clean_body'] = [re.sub(r"((?:https?:\/\/)(?:\w[^\)\]\s]*))",'', x) for x in df['body']]

#NOTE: consider internal hyphen as full words. "Technical vocabulary"

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['etc', 'however', 'there', 'also',])

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

# remove rows with less than 2 word
df = df[df['clean_body'].map(lambda x: len(str(x).strip().split())) >= 2]

# delete RemindMeBot  
df = df[df['author'] != 'RemindMeBot']

# write the csv file
df.to_csv('preprocessed_comments.csv', index = True)
