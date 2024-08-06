import tkinter as tk
from tkinter import ttk
import numpy as np
from joblib import load

# 加載模型
model = load('lr_model.joblib')

# 定義字典
car_dict = {'無車位': 0, '其他': 1, '塔式車位': 2, '升降機械': 3, '升降平面': 4, '坡道機械': 5, '坡道平面': 6, '一樓平面': 7}
material_dict = {"其他": 0, "ＲＣ造": 1, "鋼骨造": 2, "鋼骨鋼筋混凝土造": 3, "鋼筋混凝土造": 4}
type_dict = {"公寓(5樓含以下無電梯)": 0, "華廈(10層含以下有電梯)": 1, "住宅大樓(11層含以上有電梯)": 2, "透天厝": 3}
district_dict = {'北屯區': 0, '南屯區': 1, '西屯區': 2, '北區': 3, '西區': 4, '東區': 5, '南區': 6}  # 示例字典

# 定義預測函數
def predict():
    try:
        input_values = []
        for entry, label in zip(entries, labels):
            value = entry.get()
            if label == "車位類別":
                input_values.append(car_dict[value])
            elif label == "建材":
                input_values.append(material_dict[value])
            elif label == "建物型態":
                input_values.append(type_dict[value])
            elif label == "鄉鎮市區":
                input_values.append(district_dict[value])
            else:
                input_values.append(float(value))
                
        test = np.array(input_values).reshape(1, -1)
        case = model.predict(test)
        
        # 获取开价总价
        asking_price = input_values[6]  # 假設開價總價是第七個輸入（索引為6）
        discount_rate = (1 - case.item() / asking_price) * 100
        
        price_prediction = f"輸入這些條件後，預測成交價格為 {case.item():.2f} 萬"
        discount_rate_text = f"折價率為 {discount_rate:.2f}%"

        result_label.config(text=f"{price_prediction}\n{discount_rate_text}")
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



