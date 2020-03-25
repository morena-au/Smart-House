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
#import twokenize as ark
from spellchecker import SpellChecker
#import codecs


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

    # comment = ' '.join(comment_token_list)
    # return comment
    # free_text = df.clean_text.apply(clean_comment)
    # free_text = ' '.join(list(free_text))
    # with codecs.open("free_text.txt", "w", "utf-8") as file:
    #     file.write(free_text)

    # # NOTE: missplellings and slangs
    # misspelled = spell.unknown(comment_token_list)
    # for word in misspelled:
    #     print(word)
    #     print("="*20)
    #     print(spell.correction(word))
    #     print("="*20)
    #     print(spell.candidates(word))
    # print("ROW ENDING")
    # input()

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
    
    return comment

# Apply function to clean the comment
df['clean_text'] = df.clean_text.apply(clean_comment)

df.to_csv('sub_onetree.csv', index=False, encoding='utf-8')

# FROM HERE

# remove rows with less than 2 word
df = df[df['clean_body'].map(lambda x: len(str(x).strip().split())) >= 2]

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
