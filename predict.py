from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np

df = pd.read_csv("csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")

infections = df.loc[df['Country/Region'] == 'Germany'].values.tolist()[0][32:]
time = list(range(len(infections)))

df = pd.DataFrame({'time': time, 'infections': infections})
df['log_infections'] = np.log(df.infections)

X = np.array(df.time).reshape(-1, 1)
y = df.log_infections

model = RidgeCV().fit(X, y)

df['log_prediction'] = model.predict(X)
df['predictions'] = np.exp(model.predict(X))
tomorrow = np.exp(model.predict(np.array(time[-1]+1).reshape(1, -1))[0])

print(df[['infections', 'predictions']])
print('Score: ', model.score(X,y), ' on ', len(time), ' predictions')
print('Tomorrow there will be ', tomorrow, ' new infections!')



