import datetime
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm



def read_data(path):
    """
    Reads the data and splits it into Training set and Test set
    """
    # read the data and sort it by date
    data = pd.read_csv(path)
    data = data[["date", "Appliances"]]
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by="date")

    # select training and test data. Then drop the date column
    first_day = data["date"][0]
    last_day = first_day + datetime.timedelta(weeks=13)
    tr_set = data[(data["date"] >= first_day) & (data["date"] <= last_day)]["Appliances"]
    test_set = data[data["date"] > last_day]["Appliances"]
    data = data["Appliances"]

    return data.to_numpy(), tr_set.to_numpy(), test_set.to_numpy()

    """
    Perform predict and retrain of the model
    order_ar: order of the autoregressive (AR)
    order_ma: order of the moving average (MA)
    tr_set: training set
    ts_set: test set
    retrain: boolean, if False the model is never retrained
    err_thresh: the error threshold over which a retraining is performed
    """

def predict_and_retrain(order_ar, order_ma, tr_set, ts_set, retrain, err_thresh):

    # Instantiate ARIMA statsmodels
    model = ARIMA(endog=tr_set, order=(order_ar, 0, order_ma)) #It works like ARMA when d = 0
    res = model.fit() #Fit the model

    idx_retrain = 0
    count_no_retrain = 1
    predictions_arr = []

    # Start forecasting on the test set and retrain when needed
    for i in tqdm(range(len(ts_set))):
        predictions_arr.append(res.forecast(steps=count_no_retrain)[-1])
        err = abs(ts_set[i] - predictions_arr[-1])
        if retrain and err > err_thresh:
            idx_last_retrain = i
            count_no_retrain = 1
            tr_set = np.concatenate((tr_set, ts_set[idx_retrain: i + 1])) #Adding to the tr_set new observed data
            model = ARIMA(endog=ts_set, order=(order_ar, 0, order_ma))
            res = model.fit() #Retrain
        else:
            count_no_retrain += 1

    # MAE - compute mean absolute error
    mae = np.mean(np.abs(np.subtract(ts_set, predictions_arr)))
    out = {'mae': mae, 'predictions': predictions_arr}
    filename = str(order_ar) + "_" + str(order_ma) + "_" + str(err_thresh) + "_retrain" if retrain else "" + ".json"
    with open(filename, 'w') as outf:
        json.dump(out, outf, indent='\t')
