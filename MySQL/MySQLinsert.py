import mysql.connector
import os
from contextlib import contextmanager
import pandas as pd
import numpy as np

# Get the passowrd
with open('password.txt', 'r') as file:
    database_password = file.readline()

path = os.getcwd()

files = [i for i in os.listdir(path) if i.startswith('comments')]

# Context managers for opening and closing the database
@contextmanager
def mysql_connection():
    mydb = mysql.connector.connect(user="root",
                                   password=database_password,
                                   host="localhost", 
                                   port="3305",
                                   database="reddit_smarthome")

    mycursor = mydb.cursor()

    yield mycursor

    mydb.commit()

    mydb.close()
    mycursor.close()


data = []
# create a list of tuple for each observation
for file in files:
    tmp = pd.read_csv(file)
    # Convert to native python type
    tmp = tmp.astype('object')
    tmp['created_utc'] = pd.to_datetime(tmp['created_utc'],
                                        unit='s', utc=True                                       
                                        ).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    for i in range(len(tmp)):
        data.append(tuple(tmp.iloc[i, :]))

# remove duplicates
data_pd = pd.DataFrame(data, columns=['id', 'link_id', 'parent_id', 'created_utc', 'body', 'author',
                                      'permalink', 'score', 'subreddit'])

# remove rows with all duplicate values
data_pd.drop_duplicates(keep=False, inplace=True)

# remove rows with duplicate id
data_pd.drop_duplicates(subset = 'id', keep=False, inplace=True)

# find all nan and replece them with 0 or NULL
# It's not possible to store NaN value in a FLOAT type column in mysql

data_pd['permalink'] = data_pd['permalink'].fillna('NULL')

data_pd.isnull().values.any(axis=0) # check at least one true per row

for i in data_pd.columns:
    null_index = data_pd[data_pd[str(i)].isnull()].index.tolist()
    #data_pd.loc[60045, 'body']
    data_pd.drop(null_index, inplace=True)


# convert pd.DataFrame to a list of tuples
data = [tuple(x) for x in data_pd.to_numpy()]

# insert data to mysql
sql = "INSERT INTO reddit_comments (id, link_id, parent_id, created_utc, body, author, \
                                    permalink, score, subreddit) \
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

with mysql_connection() as mycursor:
    mycursor.executemany(sql, data)