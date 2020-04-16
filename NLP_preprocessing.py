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
from sklearn.model_selection import train_test_split
import html
import NLP_visualization as NLP_vis
#import twokenize as ark
from spellchecker import SpellChecker
import MySQL_data as data
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# get source data for further investigations
comments = data.comments
submissions = data.submissions

# import new granularity file
df = pd.read_csv(".\\DataSource_backup\\df_tree.csv", encoding='utf8')

# Drop comments from bots
bots = ['_whatbot_', '_youtubot_', 'alotabot', 'anti-gif-bot', 
'by-accident-bot', 'checks_out_bot', 'cheer_up_bot', 'clichebot9000', 'could-of-bot', 'doggobotlovesyou', 
'gifv-bot', 'gram_bot', 'haikubot-1911', 'have_bot', 'icarebot', 'image_linker_bot', 
'imguralbumbot', 'navigatorbot', 'of_have_bot', 'phonebatterylevelbot', 'remembertosmilebot', 
'robot_overloard', 'serendipitybot', 'sneakpeekbot', 
'spellingbotwithtumor', 'substitute-bot', 'thank_mr_skeltal_bot', 'thelinkfixerbot', 
'timezone_bot', 'turtle__bot', 'tweettranscriberbot', 'video_descriptbotbot', 
'video_descriptionbot', 'yourewelcome_bot', 'youtubefactsbot', 'TitleLinkHelperBot', 
'Link-Help-Bot', 'vReddit_Player_Bot', 'IrrelevantXKCD-Bot', 'GoodBot_BadBot', 'DuplicatesBot', 
'im-dad-bot', 'LinkReplyBot', 'hinkbot', 'YTubeInfoBot', 'CommonMisspellingBot', 'WikiTextBot', 
'10_LETTERS_BOT', 'Yasuo_Spelling_Bot', 'BadBotPSA', 'HA-Helper-Bot', 'TerribleJokeBot', 'Polite_Users_Bot', 
'LimbRetrieval-Bot', 'The-Worst-Bot', 'CakeDayGIFt_Bot', 'JoeBidenBot', 'HelperBot_', 'CakeDay--Bot', 'HappyNumberBot', 
'PORTMANTEAU-BOT', 'TheSwearBot', 'QuoteMe-Bot', 'FatFingerHelperBot', 'RemindMeBot', 'AlexaPlayBot', 
'SmallSubBot', 'aardBot', 'itchy_robot', 'Sub_Corrector_Bot', 'umnikos_bots', 'NoMoreMisspellingBot', 
'JeopardyQBot', 'iotaTipBot', 'ComeOnMisspellingBot', 'Bot_Metric']


bot_ids = []
for i in bots:
    for x in list(comments.loc[comments.author == i, "id"]):
        bot_ids.append(x)

for i in bot_ids:
    contain = df.tree_ids.str.contains(i).unique()
    if len(contain) > 1:
        row = df.index[df.tree_ids.str.contains(i)].to_list()[0]
        id_list = re.split(r"\s(<.*?>)\s", df.iloc[row, 0])
        # remove corresponding body
        body_list = re.split(r"\s(<.*?>)\s", df.iloc[row, 1])
        idx = [num for num, elem in enumerate(id_list) if elem.strip() == i][0]
        del id_list[idx-1 : idx+1]
        # update with the new id list
        df.iloc[row, 0] = ' '.join(id_list)
        del body_list[idx-1 : idx+1]
        # update with the new body list
        df.iloc[row, 1] = ' '.join(body_list)

# drop rows left without comments
df.drop(df.index[df.tree_ids == "",].tolist(), axis=0, inplace=True)

# Concatenate sub. text with text from one tree
df['text'] = df["title"].astype(str) + " <SUB> " +\
             df["selftext"].astype(str) + " <SUB> " +\
             df["tree_bodies"].astype(str)

# Cleaning up the comments
# nltk.download('stopwords')  # (run python console)
# nltk.download('wordnet')  # (run python console)
# python3 -m spacy download en  # (run in terminal)

# Extracting URLs: external links often informative, but they add unwanted noise to our NLP model
# Strip out hyperlinks and copy thme in a new column URL

# Find URL
def find_URL(comment):
    return re.findall(r'((?:https?:\/\/)(?:\w[^\)\]\s]*))', comment)

df['URL'] = df.text.apply(find_URL)

# create a colummn with the pre-processed text:
# remove URLs
df['clean_text'] = [re.sub(r"((?:https?:\/\/)(?:\w[^\)\]\s]*))",'', x) for x in df['text']]

# Unscape html formatting
df['clean_text'] =  [html.unescape(x) for x in df['clean_text']]

#NOTE: consider internal hyphen as full words. "Technical vocabulary"
# pattern = re.compile(r"\b(\w*)-(\w*)\b", re.I)
# hyphen_words = []
# for i in df.clean_text:
#     hyphen_words.append(pattern.findall(i))

df['clean_text'] = [re.sub(r"\b(\w*)-(\w*)\b", r"\g<1>_\g<2>", x) for x in df['clean_text']]


# NLTK Stop words
stop_words = stopwords.words('english')

#Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
more_stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
             "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
             "during", "each", "few", "for", "from", "further", "had", "has", "have",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
             "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
             "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me",
             "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
             "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
             "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
             "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
             "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
             "we're", "we've", "were", "what", "what's", "when", "when's", "where",
             "where's", "which", "while", "who", "who's", "whom", "why", "why's",
             "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
             "yours", "yourself", "yourselves"]

stop_words.extend(list(set(more_stopwords) - set(stop_words)) + ['etc', 'however', 'there', 'also', 'digit'])

#We specify the stemmer or lemmatizer we want to use
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem

# load default word frequency list for misspelling
spell = SpellChecker()
spell.word_frequency.load_text_file('.\\DataSource_backup\\free_text.txt')

# Remove comments where 70% words are not part of the english vocabulary
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

def clean_comment(comment, lemma=True, del_tags = ['NUM', 'PRON', 'ADV', 'DET', 'AUX', 'SCONJ', 'PART']):
    comment = re.sub(r"(<SUB>|nan|<NEW TIER>|<SAME TIER>)", "", comment)
    comment = comment.lower() # ? consider to make general the name of companies or decives
    comment = re.sub(r'&#x200B', ' ', comment) # character code for a zero-width space
    comment = re.sub(r'remindme![\w\s\W]*$', ' ', comment) # remove call to remind me bot
    comment = re.sub(r'\n', ' ', comment) # remove new line formatting
    comment = re.sub(r'(\[deleted\]|\[removed\])', '', comment)
    comment = re.sub(r"[^\w\s]", ' ', comment) # punctuation and emoji
    comment = re.sub(r'(\s_|_\s)', '', comment) # remove underscores around a words (italics)


    # detect no english comments and remove them 
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
    comment_token_list = [word for word in comment.strip().split() if word not in stop_words and len(word)>1]

    # keeps word meaning: important to infer what the topic is about
    if lemma == True:
        # Initialize spacy 'en' model
        nlp = spacy.load('en_core_web_sm')
        # https://spacy.io/api/annotation
        comment_text = nlp(' '.join(comment_token_list))
        # for token in comment_text:
        #     print(token.pos_, "\t", token)
        comment_token_list = [token.lemma_ for token in comment_text if token.pos_ not in del_tags]
    
    # harsh to the root of the word
    else:
        comment_token_list = [word_rooter(word) for word in comment_token_list]

    comment = ' '.join(comment_token_list)

    #NOTE digits within string
    
    return comment

# Apply function to clean the comment
df['clean_text'] = df.clean_text.apply(clean_comment)

#df.to_csv('sub_onetree.csv', index=False, encoding='utf-8')
# import new granularity file
#df = pd.read_csv(".\\sub_onetree.csv", encoding='utf8')

# drop rows left without comments
#df.drop(df.index[df.clean_text == "",].tolist(), axis=0, inplace=True)
df.drop(df.index[df.clean_text.isna(),].tolist(), axis=0, inplace=True)

# remove rows with less than 15 words (short observations)
df = df.loc[df['clean_text'].map(lambda x: len(str(x).strip().split())) > 15,]

# Divide the data in 80% training and 20% test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# write df
df_train.to_csv('.\\DataSource_backup\\sub_onetree_train.csv', index=False, encoding='utf-8')
df_test.to_csv('.\\DataSource_backup\\sub_onetree_test.csv', index=False, encoding='utf-8')

# Descriptive visualization
NLP_vis.freq_words(df["clean_text"], True, 50)
NLP_vis.words_count(df["clean_text"])



