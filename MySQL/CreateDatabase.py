import mysql.connector

# Create a connection

mydb = mysql.connector.connect(
  user="root",
  password="DigitalFire19!",
  host="localhost", 
  port="3305"
)

print(mydb)

# Create a database named 'reddit_smarthome'
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE reddit_smarthome")

# Check if it has been created
mycursor.execute("SHOW DATABASES")

for x in mycursor:
    print(x)