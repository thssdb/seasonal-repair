import os

from pypots.clustering import CRLI
from pypots.utils.metrics import cal_adjusted_rand_index
import numpy as np
import pandas as pd

EPOCHS = 10

dic = {"airline": 0, "bank": 1, "grid": 2, "insurance": 3, "telecom": 4}
path = "./dataPath/"

if __name__ == '__main__':
    for method in os.listdir(path):
        train_X = []
        train_y = []
        size = 500
        for file in os.listdir(path + method + "/"):
            if int(file.split("_")[-1][:-4]) < size:
                continue
            train_y.append(dic[file.split("_")[0]])
            data = pd.read_csv(path + method + "/" + file)
            y = data["value"].to_numpy()[:size]
            train_X.append(y.reshape((-1, 1)).tolist())

        # train_X = [[[1], [2], [3]], [[2], [8], [6]], [[2], [3], [4]], [[51], [815], [1951]]]
        # train_y = np.array([0, 1, 0, 2])

        n_steps = size
        n_features = 1
        n_clusters = 5

        print("Running test cases for CRLI...")
        crli = CRLI(
            n_steps=n_steps,
            n_features=n_features,
            n_clusters=n_clusters,
            n_generator_layers=2,
            rnn_hidden_size=64,
            epochs=EPOCHS,
        )
        crli.fit(train_X)

        clustering = crli.cluster(train_X)
        RI = cal_adjusted_rand_index(clustering, train_y)
        # CP = cal_cluster_purity(clustering, train_y)

        f = open(path + "cluster-results.txt", "a")
        f.write(method + " " + str(RI) + "\n")
        f.close()

    # RI
