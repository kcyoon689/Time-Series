import pandas as pd
import plotly.express as px
pd.options.plotting.backend = "plotly"

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx'
df = pd.read_excel(url)

headers = df.iloc[0]
headers[1] = 'ISE(TL)'
headers[2] = 'ISE(USD)'
print(headers)

new_df  = pd.DataFrame(df.values[1:], columns=headers)
new_df['Price'] = new_df['ISE(USD)'].cumsum()
# print(new_df.head())

# Price Diff
fig = new_df.plot(x='date', y=['ISE(USD)', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM'], title='Price Changes')
fig.show()

# Price
fig = new_df.plot(x='date', y=['Price'], title='Price')
fig.show()

# Correlation
corr_df = new_df[['ISE(USD)', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']]
corr = corr_df.corr(method = 'pearson')
# print(corr)
print("corr_df")
print(corr_df)

fig = px.imshow(corr, text_auto=True, title='Correlation Map')
fig.show()
