import pymssql
import pandas as pd
import numpy as np

# 建立與 SQL Server 的連接
conn = pymssql.connect(
    server='10.11.144.102',
    user='m07966',
    password='Ethan60616',
    database='DB_DIGITECH',
    as_dict=True
)

try:
    SQL_QUERY = '''
    WITH CTE AS (
    SELECT * FROM A_W_202212_202312 
    UNION ALL 
    SELECT * FROM A_W_202401_202407)

    SELECT date AS 開價日期, AVG(unit) AS 平均單價 
    FROM CTE
    WHERE low >= 2 AND unit <= 1000
    GROUP BY date
    ORDER BY date 
    '''

    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)

    data = cursor.fetchall()

    # 將取得的資料轉換成 DataFrame
    df = pd.DataFrame(data)

    # 將日期欄位轉換成 datetime
    df["開價日期"] = pd.to_datetime(df["開價日期"])

    # 設置日期欄位為索引
    df.set_index("開價日期", inplace=True)

    # 過濾從第 883 行開始的資料
    df = df.iloc[882:]

    # 定義重新索引的開始和結束日期
    start_date = '2018-01-01'
    end_date = '2024-07-25'

    # 建立完整的日期範圍
    index = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(index)

    # 前向填補缺失值
    df["平均單價"] = df["平均單價"].fillna(method='ffill')

finally:
    # 關閉資料庫連接
    conn.close()

# 將 DataFrame 轉換為時間序列預測的輸入特徵和標籤
def df_to_X_y(df, window_size=9, forecast_horizon=2):
    df_as_np = df["平均單價"].to_numpy()  # 確保只取數值欄位

    X = []
    y = []
    dates = []

    for i in range(len(df_as_np) - window_size - forecast_horizon + 1):  # 修改迴圈範圍
        row = df_as_np[i:i+window_size].reshape(-1, 1)  # 轉換成正確的形狀
        X.append(row.flatten())  # 展開為一維
        labels = df_as_np[i+window_size:i+window_size+forecast_horizon]  # 取兩個時間點的數值
        y.append(labels)
        dates.append(df.index[i + window_size])  # 紀錄對應的時間索引

    return np.array(X), np.array(y), dates

# 測試 df_to_X_y 函數
X, y, dates = df_to_X_y(df)

# 將 X 和 y 組合成 DataFrame，並加入時間索引
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y_df = pd.DataFrame(y, columns=[f"label_{i+1}" for i in range(y.shape[1])])

# 合併 X 和 y DataFrame
result_df = pd.concat([pd.Series(dates, name="index_date"), X_df, y_df], axis=1)
result_df.set_index("index_date", inplace=True)

print(result_df.head())