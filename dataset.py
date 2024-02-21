import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('./chinook.db')

# Get a cursor object
cur = conn.cursor()

# Query the sqlite_master table for all tables
cur.execute("SELECT * FROM sqlite_master")
tables = cur.fetchall()

# # For each table, select all data and write to a CSV file
# for table in tables:
#     table = table[0]
#     df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
#     df.to_csv(f"{table}.csv", index=False)
df = pd.read_sql_query("SELECT * FROM sqlite_master", conn)
df.to_csv("sqlite_master.csv", index=False)
# Close the connection to the database
conn.close()