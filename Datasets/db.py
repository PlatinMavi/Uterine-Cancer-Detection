import pandas as pd
import sqlite3

df = pd.read_csv('Datasets\default.csv')

db_connection = sqlite3.connect('blood.db')

df.to_sql('blood', db_connection, if_exists='replace', index=False)

db_connection.close()
