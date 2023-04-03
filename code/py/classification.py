from pypots.classification import GRUD
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

EPOCHS = 10

dic = {"airline": 0, "bank": 1, "grid": 2, "insurance": 3, "telecom": 4}
path = "./dataPath/"

if __name__ == "__main__":
    for method in os.listdir(path):
        X = []
        y = []
        size = 500
        for file in os.listdir(path + method + "/"):
            if int(file.split("_")[-1][:-4]) < size:
                continue
            y.append(dic[file.split("_")[0]])
            data = pd.read_csv(path + method + "/" + file)
            X_temp = data["value"].to_numpy()[:size]
            X.append(X_temp.reshape((-1, 1)).tolist())

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=66)

        n_steps = size
        n_features = 1
        n_clusters = 5

        print("Running test cases for GRUD...")
        grud = GRUD(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=64,
            n_classes=n_clusters,
            epochs=EPOCHS,
        )
        grud.fit(train_X, train_y, test_X, test_y)

        predictions = grud.classify(test_X).tolist()
        # predictions = grud.classify(X).tolist()

        pred_y = [i.index(max(i)) for i in predictions]
        # test_y = np.array(test_y)

        # F1-score
        f1_micro = f1_score(test_y, pred_y, average="micro")
        # f1_micro = f1_score(y, pred_y, average="micro")

        f = open(path + "classification-results.txt", "a")
        f.write(method + " " + str(f1_micro) + "\n")
        f.close()
