import sqlite3
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# Connect to the SQLite database
conn = sqlite3.connect('chinook.db')

# Query the database and load the data into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM sqlite_master", conn)


print(df)