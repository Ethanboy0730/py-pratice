# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:02:31 2024

@author: m07966
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_excel("time_series_data.xlsx", index_col="Unnamed: 0", parse_dates=True)

# 定義函數將 DataFrame 轉換為特徵和標籤
def df_to_X_y(df, window_size=28, forecast_horizon=7):
    df_as_np = df["平均單價"].to_numpy()  # 確保只取數值欄位

    X = []
    y = []
    dates = []

    for i in range(len(df_as_np) - window_size - forecast_horizon + 1):
        row = df_as_np[i:i+window_size].reshape(-1, 1)  # 轉換成正確的形狀
        X.append(row.flatten())  # 展開為一維
        labels = df_as_np[i+window_size:i+window_size+forecast_horizon]
        y.append(labels)
        dates.append(df.index[i + window_size])  # 紀錄對應的時間索引

    return np.array(X), np.array(y), dates

# 測試 df_to_X_y 函數
window_size = 28
forecast_horizon = 7
X, y, dates = df_to_X_y(df, window_size, forecast_horizon)

# 將 X 和 y 組合成 DataFrame，並加入時間索引
X_df = pd.DataFrame(X, columns=[f"t-{window_size-i}" for i in range(X.shape[1])], index=dates)
y_df = pd.DataFrame(y, columns=[f"t+{i+1}" for i in range(y.shape[1])], index=dates)

# 合併 X 和 y DataFrame
result_df = pd.concat([pd.Series(dates, name="index_date"), X_df, y_df], axis=1)
result_df.set_index("index_date", inplace=True)

# 分割訓練和測試數據
train_size = int(len(X_df) * 0.8)
X_train, X_test = X_df.iloc[:train_size], X_df.iloc[train_size:]
y_train, y_test = y_df.iloc[:train_size], y_df.iloc[train_size:]

# 訓練模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 將預測結果轉換為 DataFrame，並設置日期索引
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

# 視覺化預測結果
plt.figure(figsize=(14, 7))

# 繪製實際值和預測值
for i in range(forecast_horizon):
    plt.plot(y_test.index, y_test.iloc[:, i], label=f"Actual t+{i+1}")
    plt.plot(y_pred_df.index, y_pred_df.iloc[:, i], linestyle='--', label=f"Predicted t+{i+1}")

plt.title("Actual vs Predicted Average Unit Price")
plt.xlabel("Date")
plt.ylabel("Average Unit Price")
plt.legend()
plt.show()

# 評估模型性能
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 印出預測結果
print(y_pred_df)

