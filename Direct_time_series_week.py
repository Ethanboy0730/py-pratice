# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:27:29 2024

@author: m07966
"""
#預測未來12周
#我們用skforecast 套件跑跑看


#Direct multi-step forecaster
#%%
import pymssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,root_mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from skforecast.utils import save_forecaster
import joblib
import warnings
from skforecast.model_selection import backtesting_forecaster

warnings.filterwarnings("ignore")
#%%

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
WHERE low >=2 and unit<=1000 and city = '台中市'

GROUP BY date
ORDER BY date 
'''

cursor = conn.cursor()
cursor.execute(SQL_QUERY)

data=cursor.fetchall()


df=pd.DataFrame(data)



df["開價日期"]=pd.to_datetime(df["開價日期"])

df.set_index("開價日期",inplace=True)

df=df.iloc[800:]





# 將缺失值填充為對應月份的平均值
df['平均單價'] = df.groupby([df.index.month,df.index.year])['平均單價'].transform(lambda x: x.fillna(x.mean()))

#換成周data
df=df['平均單價']
df=df.resample('W').mean()
data=df

#%%
# Split train-test
# ==============================================================================

# Train-validation dates
# ==============================================================================
end_train = '2023-11-19 00:00:00'
start_test = '2023-11-26 00:00:00'
print(
    f"Train dates      : {data.index.min()} --- {data.loc[:end_train].index.max()}"
    f"  (n={len(data.loc[:end_train])})"
)
print(
    f"Validation dates : {data.loc[end_train:].index.min()} --- {data.index.max()}"
    f"  (n={len(data.loc[end_train:])})"
)



# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3))
data.loc[:end_train].plot(ax=ax, label='train')
data.loc[end_train:].plot(ax=ax, label='validation')
ax.legend()
plt.show()

display(data.head(4))

#%%
def custom_weights(index):
    """
    Return 0 if index is between 2021-07-25 and '2022-07-03.
    """
    weights = np.where(
                  (index >= '2021-07-25') & (index <= '2022-07-03'),
                   0,
                   1
              )

    return weights

#%%

# Create and fit forecaster
# ==============================================================================
models = [RandomForestRegressor(random_state=123), 
          GradientBoostingRegressor(random_state=123),
          Ridge(random_state=123),
          Lasso(random_state=123)]

# Hyperparameter to search for each model
param_grids = {'RandomForestRegressor': {'n_estimators': [10,50, 100], 'max_depth': [5,10,15]},
               'GradientBoostingRegressor': {'n_estimators': [10,20, 50], 'max_depth': [5,10,15]},
               'Ridge': {'alpha': [0.01, 0.1, 1]},
               'Lasso': {'alpha': [0.01, 0.1, 1]}}

# Lags used as predictors
lags_grid = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

df_results = pd.DataFrame()
for i, model in enumerate(models):

    print(f"Grid search for regressor: {model}")
    print(f"-------------------------")

    forecaster = ForecasterAutoreg(
                     regressor = model,
                     lags      = 3,
                     weight_func=custom_weights
                 )

    # Regressor hyperparameters
    param_grid = param_grids[list(param_grids)[i]]

    results_grid = grid_search_forecaster(
                       forecaster         = forecaster,
                       y                  = data,
                       param_grid         = param_grid,
                       lags_grid          = lags_grid,
                       steps              = 4,
                       refit              = 3,
                       metric             = 'mean_squared_error',
                       initial_train_size = len(data.loc[:end_train]),
                       fixed_train_size   = True,
                       return_best        = True,
                       n_jobs             = 'auto',
                       verbose            = False,
                       show_progress      = True
                   )
    
    # Create a column with model name
    results_grid['model'] = list(param_grids)[i]
    
    df_results = pd.concat([df_results, results_grid])

df_results = df_results.sort_values(by='mean_squared_error')
df_results.head(10)


#%%

forecaster_Rf = ForecasterAutoreg(
                 regressor = Ridge(alpha=0.01,random_state=123),
                 lags      = 23,
                 weight_func=custom_weights
             )

metric, predictions = backtesting_forecaster(
                          forecaster            = forecaster,
                          y                     = data,
                          steps                 = 4,
                          metric                = 'mean_squared_error',
                          initial_train_size    = len(data.loc[:end_train]),
                          fixed_train_size      = False,
                          allow_incomplete_fold = True,
                          refit                 = 3,
                          verbose               = True,
                          show_progress         = True  
                      )


# Plot predictions
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3))
data.loc[start_test:].plot(ax=ax)
predictions.plot(ax=ax)
ax.legend();
#%

#%%


#算每4周趨勢做一個比對
# 計算每4周的趨勢
predictions_trend = predictions.rolling(window=4).mean()








#%%
# Create and train forecaster
forecaster_Rf = ForecasterAutoreg(
                 regressor     = RandomForestRegressor(random_state=123),
                 lags          = 5,
                 forecaster_id = "forecaster_rf"
             )

forecaster_Rf.fit(y=data)

forecaster_rf=joblib.dump(forecaster_Rf,'forecaster_rf.joblib')


#%%

forecaster_Ridge = ForecasterAutoreg(
                 regressor     = Ridge(random_state=123,alpha=0.01),
                 lags          = 23,
                 forecaster_id = "forecaster_Ridge"
             )

forecaster_Ridge.fit(y=data)

forecaster_ridge=joblib.dump(forecaster_Ridge,'forecaster_ridge.joblib')


#Backtesting with intermittent refit


