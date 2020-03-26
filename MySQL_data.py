# Get the passowrd
from contextlib import contextmanager
import numpy as np
import pandas as pd
import mysql.connector

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