import matplotlib.pyplot as plt
import json
import copy
from utility import read_data, predict_and_retrain
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller # Augmented Dickey Fuller test
import numpy as np

# Load data

path_1 = 'energydata_complete.csv'
path_2 = '_trend_energydata_complete.csv'

data , tr_set, ts_set = read_data(path_1)
data_trend , tr_set_trend, ts_set_trend = read_data(path_2)

#-------------

total_duration = len(data)
step = 1
time = np.arange(0, total_duration, step)

# Original time seires
plt.plot(time, data)
plt.show()

# Time series with trend
plt.plot(time, data_trend)
plt.show()

#-------------

# Augmented Dickey-Fuller test
# (to provide a quick check and confirmatory evidence that the time series is stationary or non-stationary)
adf = adfuller(data)
adf, pvalue, critical_values = adf[0], adf[1], adf[4]
print(f"ADF: {adf}\np-value: {pvalue}")
print("Critical values:")
for k, v in critical_values.items():
    print(f"{k}: {v}")

adf2 = adfuller(data_trend)
adf2, pvalue2, critical_values2 = adf2[0], adf2[1], adf2[4]
print(f"ADF: {adf2}\np-value: {pvalue2}")
print("Critical values:")
for k2, v2 in critical_values2.items():
    print(f"{k2}: {v2}")

#-------------


# Plot autocorrelation and partial autocorrelation
fig, ax = plt.subplots(1,2,figsize=(15,5))
plot_pacf(tr_set, lags=30, ax=ax[0])
plot_acf(tr_set, lags=20, ax=ax[1])
plt.show()

fig, ax = plt.subplots(1,2,figsize=(15,5))
plot_pacf(tr_set_trend, lags=30, ax=ax[0])
plot_acf(tr_set_trend, lags=20, ax=ax[1])
plt.show()

#-------------

# Training models

model_order = {"order_ar": 3, "order_ma": 0, "err_thresh": 0}

predict_and_retrain(**model_order, tr_set=copy.deepcopy(tr_set), ts_set=ts_set, retrain=True)

predict_and_retrain(**model_order, tr_set=copy.deepcopy(tr_set_trend), ts_set=ts_set_trend, retrain=True)

#-------------

# Results
models_data = {}
thresh_to_idx = {0: 0, 1: 2}
best_models = {0: "3_0_0_retrain(no_Trend)", 2: "3_0_0_retrain(Trend)"}
for thresh, filename in best_models.items():
    with open(filename) as f:
        models_data[thresh] = json.load(f)

n = 24 * 6
fig, axs = plt.subplots(2)
axs[0].plot(ts_set[-n:], label="Test data")
axs[0].plot(models_data[thresh_to_idx[0]]["predictions"][-n:], label="Predictions", color='#d35400')
axs[0].set_title("AR(3) - Retrain after every prediction (stationary time series)")
axs[0].legend()

axs[1].plot(ts_set_trend[-n:], label="Test data")
axs[1].plot(models_data[thresh_to_idx[1]]["predictions"][-n:], label="Predictions", color='#e74c3c')
axs[1].set_title("AR(3) - Retrain after every prediction (non-stationary time series)")
axs[1].legend()
plt.show()