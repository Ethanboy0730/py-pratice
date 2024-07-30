# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:32:10 2024

@author: m07966
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:23:31 2024

@author: m07966
"""

import pymssql
import pandas as pd
import numpy as np


conn = pymssql.connect(
    server='10.11.144.102',
    user='m07966',
    password='Ethan60616',
    database='DB_DIGITECH',
    as_dict=True
)

SQL_QUERY='''

WITH CTE AS (
SELECT * FROM A_W_202212_202312 
UNION ALL 
SELECT * FROM A_W_202401_202407)

SELECT date AS 開價日期,AVG(unit) AS 平均單價 FROM CTE
WHERE low >=2 and unit<=1000

GROUP BY date
ORDER BY date 
'''

cursor = conn.cursor()
cursor.execute(SQL_QUERY)

data=cursor.fetchall()


df=pd.DataFrame(data)


df["開價日期"]=pd.to_datetime(df["開價日期"])

df.set_index("開價日期",inplace=True)

df=df.iloc[882:]

start_date = '2018-01-01'
end_date = '2024-07-25'

index = pd.date_range(start=start_date, end=end_date, freq='D')
df=df.reindex(index)

df["平均單價"]=df["平均單價"].fillna(method='ffill')



#重頭戲
def df_to_X_y(df, window_size=9):
    df_as_np = df.to_numpy()

    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        labels = df_as_np[i+window_size:i+window_size+2]
        y.append(labels)

    return np.array(X), np.array(y)
    


WINDOW_SIZE = 9
x, y = df_to_X_y(df['平均單價'], WINDOW_SIZE)


#%%
#%%

df_X = pd.DataFrame(x.reshape(len(x), -1), columns=[f'X_{i+1}' for i in range(WINDOW_SIZE)])
df_y = pd.DataFrame(y, columns=['y'])
df_combined = pd.concat([df_X, df_y], axis=1)

#%%

start_date = '2018-01-08'
end_date = '2024-07-25'
index = pd.date_range(start=start_date, end=end_date, freq='D')
df_combined["date"]=index

df_combined=df_combined.set_index("date")

#%%

df_combined.to_csv("time_series.csv")