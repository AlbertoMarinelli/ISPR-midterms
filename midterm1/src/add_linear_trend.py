import numpy as np
import pandas as pd
import datetime

# read the data and sort it by date

data = pd.read_csv("energydata_complete.csv")
data = data[["date", "Appliances"]]
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(by="date")

total_duration = len(data)
step = 1
time = np.arange(0, total_duration, step)

k0= 0.01
k1= 0.5

series_trend = k0 + k1 * time
past_data = data["Appliances"]
data["Appliances"]= past_data + series_trend


data.to_csv(r'_trend_energydata_complete.csv')