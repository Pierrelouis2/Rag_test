import sqlite3
from prettytable import PrettyTable

#Connect to SQLite database
conn = sqlite3.connect('chinook.db')

# Create a cursor
cursor = conn.cursor()

# Execute a query
cursor.execute("""
 SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'index');
""")

# cursor.execute("""
#  SELECT p.country, MAX(pt.playtime) AS max_playtime FROM playlists p JOIN playlist_track pt ON p.id = pt.playlist_id GROUP BY p.country;

# """)

# Fetch all rows from the last executed statement
rows = cursor.fetchall()
# print(cursor.description)
print(rows)
for row in rows:
    print(row)
    print("\n")

# x = PrettyTable()

# # Set the column names
# x.field_names = [desc[0] for desc in cursor.description]

# # Add rows
# for row in rows:
#     x.add_row([row[col] for col in x.field_names])

# print(x)
# Commit changes and close connection
conn.commit()
conn.close()