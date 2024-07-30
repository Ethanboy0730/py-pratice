# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:02:31 2024

@author: m07966
"""

import pandas as pd
import numpy as np


df=pd.read_excel("time_series_data.xlsx",index_col="Unnamed: 0",parse_dates=True)


def df_to_X_y(df, window_size=28, forecast_horizon=1):
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
window_size = 28
forecast_horizon = 7
X, y, dates = df_to_X_y(df, window_size, forecast_horizon)

# 將 X 和 y 組合成 DataFrame，並加入時間索引
X_df = pd.DataFrame(X, columns=[f"t-{window_size-i}" for i in range(X.shape[1])], index=dates)  # 修改欄位名稱和加入時間索引
y_df = pd.DataFrame(y, columns=[f"t+{i}" for i in range(y.shape[1])], index=dates)

# 合併 X 和 y DataFrame
result_df = pd.concat([pd.Series(dates, name="index_date"), X_df, y_df], axis=1)
result_df.set_index("index_date", inplace=True)



X_train=X_df.iloc[:2274]
y_train=y_df.iloc[:2274]

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()


model.fit(X_train,y_train)

# 設定初始輸入資料為最後一個窗口的資料
input_data=X_train.iloc[-1].values

# 設定預測的時間長度
prediction_length = 30

# 進行無限遞迴的預測
predictions = []
for _ in range(prediction_length):
    # 進行單一時間點的預測
    prediction = model.predict(input_data)
    predictions.append(prediction[0])  # 取得預測結果的第一個時間點值

    # 更新輸入資料，將預測結果加入到最後一個窗口的資料中
    input_data = np.concatenate((input_data[:, 1:], prediction), axis=1)

# 將預測結果轉換為 DataFrame
predictions_df = pd.DataFrame(predictions, columns=y_train.columns)

# 印出預測結果
print(predictions_df)


