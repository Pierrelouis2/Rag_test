import sqlite3

#Connect to SQLite database
conn = sqlite3.connect('chinook.db')

# Create a cursor
cursor = conn.cursor()

# Execute a query
cursor.execute("SELECT * FROM tracks where AlbumId = 1")

# Fetch all rows from the last executed statement
rows = cursor.fetchall()

for row in rows:
    print(row)

# Commit changes and close connection
conn.commit()
conn.close()