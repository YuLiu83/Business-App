import sqlite3
import os
import pandas as pd

if os.path.exists('database.sqlite'):
    os.remove('database.sqlite')
conn=sqlite3.connect('database.sqlite')
c = conn.cursor()

#c.execute(''' create table clients ([Label] INTEGER , [Text] text)''')

read_data=pd.read_csv(r'data.csv')
read_data.to_sql('Text_Data', conn, if_exists='append', index=False)

c.execute('''select * from Text_Data''').fetchall()

#conn.commit()

x=pd.read_sql('''select * from Text_Data''', conn)
print(x.head())



# https://datatofish.com/create-database-python-using-sqlite3/
# https://mungingdata.com/sqlite/create-database-load-csv-python/