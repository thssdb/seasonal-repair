# from darts.models.forecasting.dlinear import DLinearModel
import random

import pandas as pd
from darts.models.forecasting.nlinear import NLinearModel

from darts import TimeSeries
from darts.metrics.metrics import rmse
import numpy as np
import os

path = "./input/ds_task/"


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


if __name__ == '__main__':
    for method in os.listdir(path):
        for file in os.listdir(path + method + "/"):
            df = pd.read_csv(path + method + "/" + file)
            y = standardization(df["value"].to_numpy())
            # random.seed(666)
            # for i in range(3441 // 5 * 4):
            #     if random.randint(1, 5) <= 3:
            #         y[i] += random.random() * 5
            data = TimeSeries.from_values(y)
            td_train, td_test = data.split_before(0.8)

            input_chunk_length = 288 * 2
            output_chunk_length = 288
            test_len = len(td_test)

            model = NLinearModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=10,
                random_state=66,
                normalize=True
            )
            model.fit(td_train)
            td_pred = model.predict(n=test_len)

            res = rmse(td_test, td_pred)

            f = open(path + "forecasting-results.txt", "a")
            f.write(method + " " + file.split("_")[0] + " " + file.split("_")[2].split(".")[0] + " " + str(res) + "\n")
            f.close()

    # RMSE
