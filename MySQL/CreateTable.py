import mysql.connector
from contextlib import contextmanager

# Get the passowrd
with open('password.txt', 'r') as file:
    database_password = file.readline()

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


# use DATETIME instead of timestamp in order to avoid DST (daylight saving time)
with mysql_connection() as mycursor:
    mycursor.execute("CREATE TABLE reddit_comments(id VARCHAR(255), link_id VARCHAR(255),\
                                                  parent_id VARCHAR(255), created_utc DATETIME,\
                                                  body TEXT, author VARCHAR(255), \
                                                  permalink VARCHAR(255), score INT, subreddit VARCHAR(255))")


# Define primary key a column with a unique key for each record¨
with mysql_connection() as mycursor:
    mycursor.execute("ALTER TABLE reddit_comments ADD PRIMARY KEY (id)")

with mysql_connection() as mycursor:
    mycursor.execute("CREATE TABLE reddit_submissions(id VARCHAR(255) PRIMARY KEY,\
                                                      created_utc DATETIME,\
                                                      title TEXT, selftext TEXT, author VARCHAR(255), \
                                                      permalink VARCHAR(255), score INT, subreddit VARCHAR(255),\
                                                      num_comments INT)")
with mysql_connection() as mycursor:
    mycursor.execute("ALTER TABLE reddit_submissions \
       ADD link_id VARCHAR(255)")



with mysql_connection() as mycursor:
    mycursor.execute("SHOW TABLES")
    for x in mycursor:
        print(x)



# EXTRA COMMANDS 
# sql = "SHOW COLUMNS FROM reddit_comments"
# sql = "ALTER TABLE reddit_comments DROP COLUMN category"
# sql = "DROP TABLE reddit_submissions"
# mycursor.execute(sql)
# for x in mycursor:
#     print(x)