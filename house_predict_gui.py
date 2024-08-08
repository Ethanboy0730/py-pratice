# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:26:16 2024

@author: m07966
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from joblib import load
import pkg_resources
import os
import sys
import sklearn
from sklearn.linear_model import LinearRegression

os.chdir('C:/Users/m07966/Desktop/m07966/折價綠')




# 加載模型

model_path='lr_model_2.joblib'
    
model=load(model_path)

# 定義字典

car_dict = {'無車位': 0, '其他': 1, '塔式車位': 2, '升降機械': 3, '升降平面': 4, '坡道機械': 5, '坡道平面': 6, '一樓平面': 7}
material_dict = {"其他": 0, "ＲＣ造": 1, "鋼骨造": 2, "鋼骨鋼筋混凝土造": 3, "鋼筋混凝土造": 4}
type_dict = {"公寓(5樓含以下無電梯)": 0, "華廈(10層含以下有電梯)": 1, "住宅大樓(11層含以上有電梯)": 2, "透天厝": 3}
district_dict = {'中區': 0, '北區': 1, '北屯區': 2, '南區': 3, '南屯區': 4, '后里區': 5, '外埔區': 6, '大甲區': 7, '大肚區': 8, '大里區': 9, '大雅區': 10, '太平區': 11, '新社區': 12, '東勢區': 13, '東區': 14, '梧棲區': 15, '沙鹿區': 16, '清水區': 17, '潭子區': 18, '烏日區': 19, '神岡區': 20, '西區': 21, '西屯區': 22, '豐原區': 23, '霧峰區': 24, '龍井區': 25}
admin_dict={"無":0,"有":1}
date_dict={'2023Q1': 0, '2023Q2': 1, '2023Q3': 2, '2023Q4': 3, '2024Q1': 4, '2024Q2': 5}

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
            elif label == "有無管理組織":
                input_values.append(admin_dict[value])
            elif label == "刊登季度":
                input_values.append(date_dict[value])
           
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
labels = ['鄉鎮市區', '開價總樓層', '開價移轉層次', '建物型態', '開價屋齡', '開價面積', '開價總價', '開價單價',
       '建物現況格局-房', '車位類別', '有無管理組織', '建材', '刊登季度']

entries = []
for label in labels:
    frame = ttk.Frame(root)
    frame.pack(fill='x')
    
    ttk.Label(frame, text=label, width=20).pack(side='left')
    
    # 根據標籤決定是創建 Entry 還是 Combobox
    if label in ["車位類別", "建材", "建物型態", "鄉鎮市區",'有無管理組織','刊登季度']:
        # 使用下拉選單
        combobox = ttk.Combobox(frame, values=list(car_dict.keys() if label == "車位類別" else
                                                   material_dict.keys() if label == "建材" else
                                                   type_dict.keys() if label == "建物型態" else
                                                   district_dict.keys() if label == "鄉鎮市區" else
                                                   admin_dict.keys() if label == "有無管理組織" else
                                                   date_dict.keys()))
        combobox.pack(fill='x', expand=True)
        entries.append(combobox)
    else:
        # 使用輸入框
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