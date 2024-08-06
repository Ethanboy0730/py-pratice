import tkinter as tk
from tkinter import ttk
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# 加載模型
model = load('lr_model.joblib')

# 定義預測函數
def predict():
    try:
        input_values = [float(entry.get()) for entry in entries]
        test = np.array(input_values).reshape(1, -1)
        case = model.predict(test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        price_prediction = f"開價1600萬，成交價預測{case.item():.2f}萬，折價率預測為{(1 - case.item() / 1600) * 100:.2f}%"
        price_error_range = f"價格誤差範圍介於{(case.item() - rmse):.2f}萬，{(case.item() + rmse):.2f}萬"
        discount_rate_range = f"折價率誤差範圍介於{(1 - (case.item() - rmse) / 1600) * 100:.2f}%，{(1 - (case.item() + rmse) / 1600) * 100:.2f}%"
        discount_rate_error = "折價率誤差正負2.92%"
        avg_price_error = f"預測價格實際價格平均會差{round(mape * 100, 2)}%"

        result_label.config(text=f"{price_prediction}\n{price_error_range}\n{discount_rate_range}\n{discount_rate_error}\n{avg_price_error}")
    except Exception as e:
        result_label.config(text=f"錯誤: {e}")

# 初始化 GUI
root = tk.Tk()
root.title("房價預測")

# 創建輸入框
labels = ["鄉鎮市區", "開價總樓層", "開價移轉層次", "建物型態", "開價屋齡", "開價面積", "開價總價", "開價單價",
          "房型", "車位類別", "銷售天期", "建材", "有無管理組織_無", "開價季度_2023Q4", "開價季度_2024Q1", "開價季度_2024Q2"]

entries = []
for label in labels:
    frame = ttk.Frame(root)
    frame.pack(fill='x')
    ttk.Label(frame, text=label, width=20).pack(side='left')
    entry = ttk.Entry(frame)
    entry.pack(fill='x', expand=True)
    entries.append(entry)

# 創建預測按鈕
predict_button = ttk.Button(root, text="預測", command=predict)
predict_button.pack(pady=10)

# 顯示結果的標籤
result_label = ttk.Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=10)

# 運行 GUI
root.mainloop()
