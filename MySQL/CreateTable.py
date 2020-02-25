import mysql.connector

# Get the passowrd
with open('password.txt', 'r') as file:
    database_password = file.readline()

mydb = mysql.connector.connect(
  user="root",
  password=database_password,
  host="localhost", 
  port="3305",
  database="reddit_smarthome"
)

mycursor = mydb.cursor()

# use DATETIME instead of timestamp in order to avoid DST (daylight saving time)
mycursor.execute("CREATE TABLE reddit_comments(id VARCHAR(255), link_id VARCHAR(255),\
                                               parent_id VARCHAR(255), created_utc DATETIME,\
                                               body TEXT, author VARCHAR(255), \
                                               permalink VARCHAR(255), score INT, subreddit VARCHAR(255),\
                                               category VARCHAR(255))")

# Define primary key a column with a unique key for each record¨
mycursor.execute("ALTER TABLE reddit_comments ADD PRIMARY KEY (id)")


mycursor.execute("CREATE TABLE reddit_submissions(link_id VARCHAR(255) PRIMARY KEY,\
                                                  created_utc DATETIME,\
                                                  body TEXT, author VARCHAR(255), \
                                                  permalink VARCHAR(255), score INT, subreddit VARCHAR(255),\
                                                  num_comments INT)")

mycursor.execute("SHOW TABLES")
for x in mycursor:
    print(x)