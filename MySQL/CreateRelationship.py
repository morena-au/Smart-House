import mysql.connector
import os
from contextlib import contextmanager
import pandas as pd
import numpy as np

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

# delete orphan comments: tot 404
# current comments shape: 337043
# Delete sql rows where IDs do not have a match from another table

# sql = 'TRUNCATE TABLE reddit_comments'
comments = pd.read_csv('G:\\SmartHome\\DataSource_backup\\reddit_comments.csv')

comments.drop(comments[comments.link_id.isin(set(comments['link_id']) - 
                                             set(submissions['link_id']))].index, inplace=True)
comments['permalink'] = comments['permalink'].fillna('NULL')

data = [tuple(x) for x in comments.to_numpy()]

# Other comments
sql = "INSERT INTO reddit_comments (id, link_id, parent_id, created_utc, body, author, \
                                    permalink, score, subreddit) \
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"


with mysql_connection() as mycursor:
    mycursor.executemany(sql, data)

# Reference the two tables
with mysql_connection() as mycursor:
    mycursor.execute("ALTER TABLE reddit_submissions DROP PRIMARY KEY")

with mysql_connection() as mycursor:
    mycursor.execute("ALTER TABLE reddit_submissions ADD PRIMARY KEY (link_id)")


sql = "ALTER TABLE reddit_comments \
            ADD CONSTRAINT FK_SubmissionComments \
            FOREIGN KEY (link_id) REFERENCES reddit_submissions (link_id)"

with mysql_connection() as mycursor:
    mycursor.execute(sql)


# comments without submissions
# I expect them to be at the beginning or at the end of the data extractions
#foo = comments[comments['link_id'].isin(set(comments['link_id']) - set(submissions['link_id']))]
#foo['created_utc'] = foo['created_utc'].astype('datetime64')
#import matplotlib as plt
#foo['created_utc'].groupby([foo['created_utc'].dt.year, foo['created_utc'].dt.month]).count().plot(kind='bar')
#plt.show()