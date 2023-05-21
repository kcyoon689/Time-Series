import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import trange
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
pd.options.plotting.backend = "plotly"

def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx'
df = pd.read_excel(url)

headers = df.iloc[0]
headers[1] = 'ISE(TL)'
headers[2] = 'ISE(USD)'
print(headers)

new_df  = pd.DataFrame(df.values[1:], columns=headers)
new_df['Price'] = new_df['ISE(USD)'].cumsum()
print(new_df.head())

# Price
split_rate = 0.8
train_df, test_df = new_df[0:int(len(new_df) * split_rate)], new_df[int(len(new_df) * split_rate):]

train_ar = train_df['Price'].values
test_ar = test_df['Price'].values

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df['date'], y=train_df['Price'],
                    mode='lines',
                    name='train'))
fig.add_trace(go.Scatter(x=test_df['date'], y=test_df['Price'],
                    mode='lines',
                    name='test'))

fig.update_layout(title='Dataset',
                   xaxis_title='Date',
                   yaxis_title='ISE(USD)')

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
fig.add_trace(go.Scatter(x=train_df['date'], y=train_df['Price'],
                    mode='lines',
                    name='train'))
fig.add_trace(go.Scatter(x=test_df['date'], y=test_df['Price'],
                    mode='lines',
                    name='test'))
fig.add_trace(go.Scatter(x=test_df['date'], y=predictions,
                    mode='lines+markers',
                    name='predictions'))

fig.update_layout(title='Training Result (arima)',
                   xaxis_title='Date',
                   yaxis_title='ISE(USD)')

fig.show()
