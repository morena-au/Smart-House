# run in shell followed by tables name: submissions or comments
# manually spot potential bots > return a list

import numpy as np
import mysql.connector
from contextlib import contextmanager
import pandas as pd
from sys import argv
import re

# unpacks argv
# pylint: disable=unbalanced-tuple-unpacking
_, table = argv


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

if table == "comments":
    # import data
    with mysql_connection() as mycursor:
        mycursor.execute('SELECT * FROM reddit_comments')
        comments = mycursor.fetchall()

    comments = pd.DataFrame(np.array(comments), 
                            columns=['id', 'link_id', 'parent_id', 'created_utc',\
                                    'body', 'author', 'permalink', 'score',\
                                    'subreddit'])

    possible_bots = np.sort(comments.loc[comments.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

    bots = []
    for i in possible_bots:
        print(i)
        print(list(comments.loc[comments.author == i, "body"][:3]))
        print("="*50)
        while True:
            try: 
                status = input("Is this a bot? yes/no ")
                if status not in ["yes", "no"]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid. Type 'yes' or 'no'")
        if status == "yes":
            bots.append(i)

    print(bots)

# # Manual screen
# ['_whatbot_', '_youtubot_', 'alotabot', 'anti-gif-bot', 
# 'by-accident-bot', 'checks_out_bot', 'cheer_up_bot', 'clichebot9000', 'could-of-bot', 'doggobotlovesyou', 
# 'gifv-bot', 'gram_bot', 'haikubot-1911', 'have_bot', 'icarebot', 'image_linker_bot', 
# 'imguralbumbot', 'navigatorbot', 'of_have_bot', 'phonebatterylevelbot', 'remembertosmilebot', 
# 'robot_overloard', 'serendipitybot', 'sneakpeekbot', 
# 'spellingbotwithtumor', 'substitute-bot', 'thank_mr_skeltal_bot', 'thelinkfixerbot', 
# 'timezone_bot', 'turtle__bot', 'tweettranscriberbot', 'video_descriptbotbot', 
# 'video_descriptionbot', 'yourewelcome_bot', 'youtubefactsbot']

if table == "submissions":
    # import data
    with mysql_connection() as mycursor:
        mycursor.execute('SELECT * FROM reddit_submissions')
        submissions = mycursor.fetchall()

    submissions = pd.DataFrame(np.array(submissions), 
                            columns=['id', 'created_utc',\
                                    'title', 'selftext', 'author', 'permalink', 'score',\
                                    'subreddit', 'num_comments', 'link_id'])

    possible_bots = np.sort(submissions.loc[submissions.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

    bots = []
    for i in possible_bots:
        print(i)
        print(list(submissions.loc[submissions.author == i, "title"][:3]))
        print(list(submissions.loc[submissions.author == i, "selftext"][:3]))
        print("="*50)
        while True:
            try: 
                status = input("Is this a bot? yes/no ")
                if status not in ["yes", "no"]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid. Type 'yes' or 'no'")
        if status == "yes":
            bots.append(i)

    print(bots)



