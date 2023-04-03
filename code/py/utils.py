import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from darts.models import RNNModel
from darts import TimeSeries
from statsmodels.tsa.arima.model import ARIMA


def lstm_ext(data_ext, interval):
    train_data = data_ext[interval:len(data_ext) - interval]
    x = TimeSeries.from_values(train_data)
    model = RNNModel(model="LSTM", input_chunk_length=10, n_epochs=10)
    model.fit(x)
    res = model.predict(n=interval).values().reshape(-1)
    for res_idx, data_idx in enumerate(range(len(data_ext) - interval, len(data_ext))):
        data_ext[data_idx] = res[res_idx]

    train_data = data_ext[len(data_ext) - interval - 1:interval - 1:-1]
    x = TimeSeries.from_values(train_data)
    model = RNNModel(model="LSTM", input_chunk_length=10, n_epochs=10)
    model.fit(x)
    res = model.predict(n=interval).values().reshape(-1)
    for res_idx, data_idx in enumerate(range(interval - 1, -1, -1)):
        data_ext[data_idx] = res[res_idx]

    return data_ext


def arima_ext(data_ext, interval):
    train_data = np.array(data_ext[interval:len(data_ext) - interval])
    model = ARIMA(train_data, order=(1, 2, 1)).fit()
    res = model.forecast(interval)
    for res_idx, data_idx in enumerate(range(len(data_ext) - interval, len(data_ext))):
        data_ext[data_idx] = res[res_idx]

    train_data = np.array(data_ext[len(data_ext) - interval - 1:interval - 1:-1])
    model = ARIMA(train_data, order=(1, 2, 1)).fit()
    res = model.forecast(interval)
    for res_idx, data_idx in enumerate(range(interval - 1, -1, -1)):
        data_ext[data_idx] = res[res_idx]

    return data_ext


def ar_ext(data_ext, interval, lags=1):
    ar = AutoReg(data_ext[interval:len(data_ext) - interval], lags=lags, old_names=False).fit().params
    for i in np.arange(interval - 1, -1, -1):  # head nan
        data_ext[i] = data_ext[i + lags] - ar[0]
        for ind, j in enumerate(np.arange(i + 1, i + lags)):
            data_ext[i] -= ar[ind + 2] * data_ext[j]
        data_ext[i] /= ar[1]
    for i in range(len(data_ext) - interval, len(data_ext)):  # tail nan
        data_ext[i] = ar[0]
        for ind, j in enumerate(np.arange(i - lags, i)):
            data_ext[i] += data_ext[j] * ar[ind + 1]
    return data_ext


def robust_decompose(s, period):
    # step 1: get trend
    trend = np.array([])
    interval = period // 2
    if period % 2 == 1:
        # head interval
        for i in range(interval):
            trend = np.append(trend, np.NaN)

        # moving median
        for i in range(len(s) - period + 1):
            trend = np.append(trend, np.median(s[i:i + period]))

        # tail interval
        for i in range(interval):
            trend = np.append(trend, np.NaN)
    else:
        # head interval
        for i in range(interval):
            trend = np.append(trend, np.NaN)

        # moving median
        for i in range(len(s) - period):
            temp = np.append(s[i + 1:i + period], (s[i] + s[i + period]) / 2)
            trend = np.append(trend, np.median(temp))

        # tail interval
        for i in range(interval):
            trend = np.append(trend, np.NaN)

    # step 1.5: fillna linear
    trend = ar_ext(trend, interval)

    # trend = lstm_ext(trend, interval)

    # step 2: de-trend
    detrend = s - trend

    # step3: seasonal
    seasonal = np.array([])
    temp = [[] for _ in range(period)]
    for i in range(len(detrend)):
        temp[i % period].append(detrend[i])

    # mean --> median
    for t in temp:
        seasonal = np.append(seasonal, np.median(t))
    median_s = np.median(seasonal)
    seasonal = seasonal - median_s

    # extend
    len_s = len(seasonal)
    for i in range(len(s) - len_s):
        seasonal = np.append(seasonal, seasonal[i % period])

    # step 4: residual
    resid = detrend - seasonal
    return (seasonal, trend, resid)


def mad_detection(r, k):
    median = np.median(r)  # np.nanmedian(r)
    mad = np.median(np.abs(r - median))
    # print(median, mad)
    lower_limit = median - (k * mad)
    upper_limit = median + (k * mad)
    # return lower_limit, upper_limit
    return np.concatenate((np.where(r > upper_limit), np.where(r < lower_limit)), axis=1)[0]


def sigma_detection(r, k):
    mean, std = np.mean(r), np.std(r)
    lower_limit = mean - (k * std)
    upper_limit = mean + (k * std)
    return np.concatenate((np.where(r > upper_limit), np.where(r < lower_limit)), axis=1)[0]


def repair(td, period, k=3, max_iter=10):
    size = len(td)
    td_repair = td.copy()
    # modified = []
    circulate = 0
    for i in range(max_iter):
        circulate += 1

        # 2-time series decomposition
        seasonal, trend, resid = robust_decompose(td_repair, period=period)

        # 3-detect
        # modify = mad_detection(r=resid, k=k)
        median = np.median(resid)  # np.nanmedian(r)
        mad = np.median(np.abs(resid - median))
        lower_limit = median - (k * mad)
        upper_limit = median + (k * mad)

        error = []
        for j in range(size):
            if resid[j] > upper_limit or resid[j] < lower_limit:
                error.append(j)
        if len(error) == 0:
            print("End with " + str(i + 1) + " circulates")
            break

        # 4-repair
        for j in error:
            arr = []
            pidx = j % period
            for s in range(size // period):
                if s * period + pidx != j:
                    arr.append(resid[s * period + pidx])
            if pidx < size % period and pidx + (size // period) * period != j:
                arr.append(resid[pidx + (size // period) * period])
            td_repair[j] = np.median(arr) + seasonal[j] + trend[j]
            if j == 290:
                print(trend[j], seasonal[j], np.median(arr) - 100, td_repair[j] - 100)
    return td_repair
