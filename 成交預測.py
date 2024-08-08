# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:17:11 2024

@author: m07966
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import os 
os.chdir('C:/Users/m07966/Desktop/m07966/折價綠')
#%%
#資料讀取與處理
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
df=pd.read_excel("台中市開價成交比對資料.xlsx",sheet_name='比較資料')
df.drop_duplicates(subset=["成交編號"], keep="first", inplace=True)
df=df.reset_index(drop=True)
df=df[["刊登日期","成交年月","縣市","鄉鎮市區","社區名稱","開價總樓層","開價移轉層次","建物型態","開價屋齡","開價面積",'開價總價','開價單價',"建物現況格局-房","折價率","車位類別","有無管理組織","銷售天期","建材","成交單價","成交總價","成交季"]]
df=df[df["開價移轉層次"]>=2]
#缺失值處理，idea 就 缺值我填無，到時建模再用label encoding 處理
df["車位類別"]=df["車位類別"].fillna("無車位")
df=df.dropna()

#極端值處理 我們處理格局 總價 單價 面積
df.describe()

for outlier in ["開價面積","開價總價","開價單價","建物現況格局-房","成交總價","成交單價","折價率","開價屋齡"]:
    stat = df[outlier].describe()
    IQR = stat["75%"] - stat["25%"]
    upper_limit = stat["75%"] + 1.5 * IQR
    lower_limit = stat["25%"] - 1.5 * IQR
    df = df[(df[outlier] <= upper_limit) & (df[outlier] >= lower_limit)]
    
df["刊登年月"]=df["刊登日期"].dt.to_period('M')
#把刊登年月轉成刊登季資料
df["刊登季度"]=df["刊登日期"].dt.to_period('Q')

#處理多於資料
df.loc[~df['建材'].isin(['鋼筋混凝土造','鋼骨鋼筋混凝土造','鋼骨造','ＲＣ造','其他']), '建材'] = '其他'
df=df[df["銷售天期"]<=365]
df["刊登季度"]=df["刊登季度"].astype("str")
df["刊登年月"]=df["刊登年月"].astype("str")
df=df[df["刊登季度"]!="2022Q3"]
df=df[df["刊登季度"]!="2022Q4"]

df["開價移轉層次"]=df["開價移轉層次"].astype(int)
df=df.drop(["縣市","社區名稱","刊登年月","刊登日期","成交季","成交單價","折價率","成交年月","銷售天期"],axis=1)

df.info()


#%%
#做完視覺化後，我們做一下特徵工程

#先處理label encoding
#我們思考一下哪些類別是有排名依序的
#車位種類?  建材? 我們嘗試用車位和建材型態做label encoding
from sklearn.preprocessing import LabelEncoder

car_dict={'無車位':0,'其他':1,'塔式車位':2,'升降機械':3,'升降平面':4,'坡道機械':5,'坡道平面':6,'一樓平面':7}
material_dict={"其他":0,"ＲＣ造":1,"鋼骨造":2,"鋼骨鋼筋混凝土造":3,"鋼筋混凝土造":4}
type_dict={"公寓(5樓含以下無電梯)":0,"華廈(10層含以下有電梯)":1,"住宅大樓(11層含以上有電梯)":2,"透天厝":3}
admin_dict={"無":0,"有":1}

df["車位類別"]=df["車位類別"].map(car_dict)
df["建材"]=df["建材"].map(material_dict)
df["建物型態"]=df["建物型態"].map(type_dict)
df["有無管理組織"]=df["有無管理組織"].map(admin_dict)

le = LabelEncoder()
df["刊登季度"] = le.fit_transform(df["刊登季度"])
df["鄉鎮市區"] = le.fit_transform(df["鄉鎮市區"])

df.info()
#

#%%做一下特徵縮放看看result
columns1=df.columns

standard=StandardScaler()

df_scaled=standard.fit_transform(df)

df_scaled=pd.DataFrame(df_scaled,columns=columns1)


#接下來我們考慮一下pca
#試試看?

floor_features=df_scaled[["開價總樓層","開價移轉層次"]]
price_features=df_scaled[["開價總價","開價單價"]]
size_features=df_scaled[["開價面積","建物現況格局-房"]]

pca=PCA(n_components=1)
pca_floor_features=pca.fit_transform(floor_features)
pca_price_features=pca.fit_transform(price_features)
pca_size_features=pca.fit_transform(size_features)

df_scaled=df_scaled.drop(["開價總樓層","開價移轉層次","開價總價","開價單價","開價面積","建物現況格局-房"],axis=1)

df_scaled["pca樓層"]=pca_floor_features
df_scaled["pca價格"]=pca_price_features
df_scaled["pca面積"]=pca_size_features

#%%
#跑模型囉
#選三個
X= df_scaled.drop(["成交總價"],axis=1)

y=df["成交總價"]  

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=101)

pipeline_lr=Pipeline([("lr_classifier",LinearRegression())])

pipeline_dt=Pipeline([("dt_classifier",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("rf_classifier",RandomForestRegressor())])


pipeline_kn=Pipeline([("kn_classifier",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("xgb_classifier",XGBRegressor())])


pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]


pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBRegressor"}
#fit model
for pipe in pipelines:
    pipe.fit(X_train, y_train)



cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_mean_absolute_percentage_error", cv=10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))

#%%
#我們做





#用xgb regressor 做看看
XGB_model = XGBRegressor()


param_grid = {"n_estimators":[10,100,500],"max_depth": [2, 4, 6, 8, 10],"min_child_weight": [4, 5, 6]}


grid = GridSearchCV(estimator=XGB_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')

grid.fit(X_train, y_train)


best_params = grid.best_params_

print("Best Hyperparameters:", best_params)

#%%
y_pred=grid.predict(X_test)

print("R^2:",metrics.r2_score(y_test, y_pred))
print("MAE:",metrics.mean_absolute_error(y_test, y_pred))
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("MAPE",mean_absolute_percentage_error(y_test,y_pred))

sns.scatterplot(x=y_test,y=y_pred)
sns.set_style("whitegrid", {"font.sans-serif": ['Microsoft JhengHei']})
plt.xlabel('實際值')
plt.ylabel('預測值')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)

#%%
#儲存模型
from joblib import dump
dump(grid,'xgb_model.joblib')



