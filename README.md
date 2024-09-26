# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
## Date: 
## AIM:
To implement the ARMA model in Python for the daily website visitors dataset.
## ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.
## PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('/content/daily_website_visitors.csv')
print(data.head())

data['Page.Loads'] = data['Page.Loads'].str.replace(',', '').astype(float)

page_loads = data['Page.Loads'].dropna()

ar1 = np.array([1, -0.5])
ma1 = np.array([1, 0.5])
arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=len(page_loads))

ar2 = np.array([1, -0.5, 0.25])
ma2 = np.array([1, 0.4, 0.3])
arma22_process = ArmaProcess(ar2, ma2)

arma22_sample = arma22_process.generate_sample(nsample=len(page_loads))

plt.figure(figsize=(12, 6))

plot_acf(arma11_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(1,1)')

plot_pacf(arma11_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(1,1)')

plot_acf(arma22_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(2,2)')

plot_pacf(arma22_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(2,2)')

plt.tight_layout()
plt.show()

model = ARIMA(page_loads, order=(2,0,2))
fitted_model = model.fit()

fitted_model_summary = fitted_model.summary()
fitted_model_summary

residuals = fitted_model.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARMA(2,2) Model for Page Loads')
plt.show()
```
## OUTPUT:
### SIMULATED ARMA(1,1) PROCESS:
#### Partial Autocorrelation
![image](https://github.com/user-attachments/assets/5ed8b8cd-b6b9-4adb-9d1b-b572faf32c2b)

#### Autocorrelation
![image](https://github.com/user-attachments/assets/0057861c-2789-44e1-9f5d-6c311a576f82)

### SIMULATED ARMA(2,2) PROCESS:
#### Partial Autocorrelation
![image](https://github.com/user-attachments/assets/4a43578e-c85d-4b72-a372-68b3fc1e162d)

#### Autocorrelation
![image](https://github.com/user-attachments/assets/2dfe66bf-8874-424b-bd7e-685c0affbe91)

## RESULT:
Thus, a Python program is created to successfully fit the ARMA Model for the daily website visitors dataset.
