import numpy as np
import pandas as pd
import os
import wget
import zipfile
import plotly.graph_objects as go
from tqdm import trange
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
pd.options.plotting.backend = "plotly"

def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

data_files_address = 'electric_power/data'

zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
zip_address = data_files_address + '/' + zip_url.split('/')[-1]
if not os.path.isfile(zip_address):
    wget.download(zip_url, out=data_files_address)

with zipfile.ZipFile(zip_address, 'r') as zip_ref:
    zip_ref.extractall(data_files_address)

txt_address = zip_address.replace('zip', 'txt')
df = pd.read_csv(txt_address, sep = ';',
                 low_memory=False, na_values=['nan', '?'])
print(len(df))
df = df.iloc[0:1000]
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
print(df.head())

# Global_active_power
split_rate = 0.8
train_df, test_df = df[0:int(len(df) * split_rate)], df[int(len(df) * split_rate):]

train_ar = train_df['Global_active_power'].values
test_ar = test_df['Global_active_power'].values

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df['Datetime'], y=train_df['Global_active_power'],
                    mode='lines',
                    name='train'))
fig.add_trace(go.Scatter(x=test_df['Datetime'], y=test_df['Global_active_power'],
                    mode='lines',
                    name='test'))

fig.update_layout(title='Dataset',
                   xaxis_title='Datetime',
                   yaxis_title='Global Active Power')

fig.show()

# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
history = [x for x in train_ar]
print(type(history))

predictions = list()
for t in trange(len(test_ar)):
    model = sm.tsa.arima.ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))

rsq_list, rmse_list, mape_list, mae_list = [],[],[],[]

# R2
Arima_r2 = r2_score(test_ar, predictions)
# print('R2: %.3f' % Arima_r2)

#RMSE
Arima_RMSE = mean_squared_error(test_ar, predictions)
# print('Testing Mean Squared Error: %.6f' % Arima_RMSE)

#MAE
Arima_MAPE = smape_kun(test_ar, predictions)
# print('Symmetric mean absolute percentage error: %.3f' % Arima_MAPE)

# MAPE 계산
Arima_MAE = np.mean(np.abs((test_ar - predictions) / test_ar)) * 100
# print("MAE: ", Arima_MAE)

# print(model_eval(test_ar, predictions))

rsq_list.append(Arima_r2)
rmse_list.append(Arima_RMSE)
mape_list.append(Arima_MAPE)
mae_list.append(Arima_MAE)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df['Datetime'], y=train_df['Global_active_power'],
                    mode='lines',
                    name='train'))
fig.add_trace(go.Scatter(x=test_df['Datetime'], y=test_df['Global_active_power'],
                    mode='lines',
                    name='test'))
fig.add_trace(go.Scatter(x=test_df['Datetime'], y=predictions,
                    mode='lines+markers',
                    name='predictions'))

fig.update_layout(title='Training Result (arima)',
                   xaxis_title='Datetime',
                   yaxis_title='Global Active Power')

fig.show()
