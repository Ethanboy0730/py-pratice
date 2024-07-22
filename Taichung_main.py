# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:48:37 2024

@author: m07966
"""
#%%
import pandas as pd
import re 
import os
os.chdir(r"C:\Users\m07966\Desktop\m07966\路名資料")
from tqdm import tqdm
from tqdm import tqdm
import time
from splitAddress import split_Address_from_Dataframe,strQ2B,split_address2,split_address3
from trade import house_year_interval,season,ch_to_num_for_street,land_interval
from tqdm import tqdm






################################################################
### Taiwan_Total_Address.xlsx存放全台灣路名的檔案 ###############
### 因為不見得所有路都是'路街道'作為結尾，所以用比對的方式清理 ####
### 這一段程式碼的目的是將xlsx檔案整理成字典 #####################
################################################################
#%%
Road_name = pd.read_excel("Taiwan_Total_Address.xlsx",sheet_name = 0)
road_dict = {}
for i in range(len(Road_name)):
    city = Road_name['縣市'][i]
    dist = Road_name['鄉鎮市區'][i]
    road = Road_name['路名'][i]
    key = (city, dist)
    if key not in road_dict:
        road_dict[key] = []
    road_dict[key].append(road)

##########################
### 小函式 ###############
##########################

wrongYear=[] #存放'清理失敗'的年度季度
def format_date(date): #轉換'交易年月日'為'ym'
    if pd.notnull(date):
        return date.strftime('%Y%m')
    else:
        return ''

#開始
years = ['110Q1','110Q2','110Q3','110Q4','111Q1','111Q2','111Q3','111Q4','112Q1','112Q2','112Q3'
         ,'112Q4','113Q1','113Q2']
         


for year in years:
    start = time.time()
    print('================== 現在開始處理{} ================='.format(year))

    deal = pd.read_csv(f"C:/Users/m07966/Desktop/m07966/路名資料/D_B_{year}_Raw.csv",encoding="utf-8-sig") #讀取檔案
    deal.drop(0,inplace=True)
    deal = deal[~deal['交易標的'].isin(['車位', '土地'])].reset_index(drop=True)   #移除車位和土地
    deal["交易年月日"]=pd.to_numeric(deal["交易年月日"],errors='coerce')
    deal = deal[(deal['交易年月日'] >= 100000) | (deal['交易年月日'].isnull())]    #移出交易年月日的異常值
    deal.reset_index(drop=True)
          #將'土地位置建物門牌'分割成'縣市''鄉鎮市區''路道街''巷''弄'......
    #result_list = []
    #for address in deal["土地位置建物門牌"]:
        #result = split_address3(address)
        #result_list.append(result)

    #address = pd.DataFrame(result_list, columns=["區", "路道街", "巷", "段"]) 
    address = split_Address_from_Dataframe(deal) 
    low_or_high = deal['移轉層次'].apply(ch_to_num_for_street) #'移轉層次:中文字轉成數字，例如:.「地下一層」轉成「-1」
    replace_dict = {"210": "20", "310": "30", "410": "40", "510": "50", "610": "60", "710": "70"}   
    low_or_high=low_or_high.replace(replace_dict, regex=True)  
    
    #交易日和建築日轉換成日期型態，以西元為單位
    trade_date =  pd.to_datetime(pd.to_numeric(deal['交易年月日'], errors='coerce') + 19110000, format='%Y%m%d', errors='coerce') #
    build_date = pd.to_datetime(pd.to_numeric(deal['建築完成年月'], errors='coerce') + 19110000, format='%Y%m%d', errors='coerce')
        
    ym = trade_date.apply(format_date) #修改成YM
    df = pd.DataFrame(index=range(len(deal)), columns=['road']) #設定一個大小和成交資料筆數相同的空資料集，為了存放路名
        #路名處理:先找Taiwan_Total_Address是否有對應的路名，如果有就放入對應的路名，如果沒有就使用分割的資料
    for j in tqdm(range(len(deal)), desc='路名處理進度'):
        try:
            allAddress = re.search(r'區(.+)',deal['土地位置建物門牌'][j]).group(1) if re.search(r'區(.+)',deal['土地位置建物門牌'][j]) else deal['土地位置建物門牌'][j]
            city = deal['縣市'][j]
            dist = deal['鄉鎮市區'][j]
            key = (city, dist)
            road = [r for r in road_dict.get(key, []) if r in allAddress]
            df['road'][j] = road[0] if road else address.loc[j, '路街道']
        except KeyError:
            continue
      
#欄位的整理及簡單的運算

    for data in ['建物移轉總面積平方公尺','總價元']:
        deal[data]=pd.to_numeric(deal[data])
    
    deal['ym'] = ym
    deal['road'] = df['road']
    deal['sec'] = address['段']
    deal['size'] = deal['建物移轉總面積平方公尺']*0.3025
    deal['price'] = deal['總價元']/10000
    deal['unit']=deal['price'] /deal['size']
    deal['floor'] = deal['總樓層數'].apply(ch_to_num_for_street)
    deal['low_high'] = low_or_high
    deal['交易年月日'] = trade_date
    deal['建築完成年月'] = build_date
    deal['屋齡'] = ((trade_date-build_date).dt.days / 365).round(1)#
 



    output = deal[['編號', 'ym','縣市', '鄉鎮市區', '交易標的', '土地位置建物門牌','road', 'sec', '交易年月日','建築完成年月','屋齡',
     '土地移轉總面積平方公尺','size','移轉層次', 'low_high','總樓層數','floor','建物移轉總面積平方公尺', '單價元平方公尺'
     , 'unit','總價元', 'price',  '建物型態','主要建材','都市土地使用分區','非都市土地使用分區', '非都市土地使用編定', '交易筆棟數',
     '主要用途',    '建物現況格局-房', '建物現況格局-廳','建物現況格局-衛', '建物現況格局-隔間', '有無管理組織',   '車位類別',
     '車位移轉總面積平方公尺', '車位總價元', '備註', '主建物面積', '附屬建物面積', '陽台面積', '電梯', '移轉編號']] 
 
    output.to_excel(f'C:/Users/m07966/Desktop/m07966/路名資料/清理資料/D_B_{year}_clean.xlsx',sheet_name='sheet1',index=False)
 #%%
 #欄位的整理及簡單的運算

 
 
 #%%
 
 
 

 
 

 
 #%% 
 
 
 
 
 
 
 

       
   

#%%
