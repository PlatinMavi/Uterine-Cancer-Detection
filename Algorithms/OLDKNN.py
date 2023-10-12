import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings
import numpy as np 
from collections import Counter

config = {
    "n_neighbor":20,
    "accMultiplier":10, #Please use at least 10 for healthy measurements and functionality of accResultMultipler.
    "accResultMultiplier":2,
    "test_size":0.1,
    "modelAffectionLimitator":0
}

class Knn:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None

    def getDistance(self, p, q):
        p = np.array(p)
        q = np.array(q)
        
        if p.shape != q.shape:
            raise ValueError("Input arrays p and q must have the same shape.")
        
        return np.sqrt(np.sum((p - q) ** 2))

    def fit(self, X, y):
        self.X = X.values
        self.y = y.values

    def predict(self, X_test):
        y_pred = []
        
        for new_point in X_test:
            distances = []
            for i, point in enumerate(self.X):
                distance = self.getDistance(point, new_point)
                distances.append([distance, self.y[i]])
            
            categories = [category[1] for category in sorted(distances)[:self.k]]
            result = Counter(categories).most_common(1)[0][0]
            y_pred.append(result)
        
        return y_pred
    
    def CalculateAccuracy(self, machine, prediction):
        machineData = machine.values
        gotRightAcc = 0

        for index in range(len(machineData)):
            if machineData[index] == prediction[index][0]:
                gotRightAcc += 1

        return gotRightAcc/len(machineData)
    
    def TrainKnn(self,selected_features):
        selected_columns = list(selected_features)
        self.fit(X_train[selected_columns], y_train)
        y_pred = []
        for i in range(len(X_machine)):
            x_var = X_machine[selected_columns].iloc[i]
            y_pred.append(self.TestKnn(selected_features, x_var, y_machine.iloc[i]))
        accuracy = self.CalculateAccuracy(y_machine,y_pred)
        
        return accuracy

    def TestKnn(self, selected_features, x_var, y_var):
        selected_columns = list(selected_features)
        self.fit(X_train[selected_columns], y_train)
        x_var_2d = x_var[selected_columns].values.reshape(1, -1)
        y_pred = self.predict(x_var_2d)
        
        return y_pred

warnings.filterwarnings('ignore', category=UserWarning)

path = os.getcwd()+r"\Datasets\default.csv"
df = pd.read_csv(path)
df.replace("-", "0.0", inplace=True)
df.replace("----", "0.0", inplace=True)

def ConvertFloat(value):
    return float(value.replace(',', '.'))

for column in df.columns[1:]:
    df[column] = df[column].apply(ConvertFloat)

df = df[df.status != "ein"]

status = {"basitler": 0,"ein":2, "highrisk": 1, "lowrisk": 1}
yUnmapped = df['status']
y = yUnmapped.map(status)
X = df.drop(columns=['status',"NRBC"])

BestScore = 0
BestScoreSeed = 0
overallscores = []

for rs in range(1000):
    KnnAlgorithm = Knn(config["n_neighbor"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config["test_size"],)

    X_hand, X_machine, y_hand, y_machine = X_test.iloc[:len(X_test)//2], X_test.iloc[len(X_test)//2:], y_test.iloc[:len(y_test)//2], y_test.iloc[len(y_test)//2:]

    TestParameters = [
        {
            "params":["MCHC", "RDWSD", "RDWCV", "PCT", "PDW"],
            "name":"Comb-1"
        },
        {
            "params":['WBC', 'MCV', 'MCHC', 'RDWSD', 'RDWCV', 'PCT', 'PDW'],
            "name":"Comb-2"
        },
        {
            "params":['EOS', 'BASO', 'WBC', 'MONO', 'HCT', 'MCHC', 'RDWSD', 'RDWCV', 'MPV', 'PCT', 'PDW'],
            "name":"Comb-3"
        },
        # {
            # "params":['LYM', 'BASO', 'HGB', 'HCT', 'MCV', 'MCHC', 'RDWSD', 'RDWCV', 'PCT', 'PDW'],
            # "name":"Comb-4"
        # }
    ]

    all_columns = X.columns.tolist()
    columns = all_columns[:]
    columnsForAppending = ['LYM', 'BASO', 'HGB', 'HCT', 'MCV', 'MCHC', 'RDWSD', 'RDWCV', 'PCT', 'PDW']

    for col in columns:
        if "_" in col:
            columns.pop(columns.index(col))

    for cosCol in columnsForAppending:
        TestParameters.append({"params":[cosCol], "name":cosCol})

    for parameters in TestParameters:
        acc = KnnAlgorithm.TrainKnn(parameters["params"])
        parameters["acc"] = acc*config["accMultiplier"]

    gotRight = 0
    gotWrong = 0

    for i,useless in enumerate(X_hand):
        x_var = X_hand.reset_index(drop=True).iloc[i]
        y_var = y_hand.reset_index(drop=True).iloc[i]

        resultsOfTests = []
        for params in TestParameters:
            resultsOfTest = {}
            pred = KnnAlgorithm.TestKnn(params["params"],x_var,y_var)
            resultsOfTest["accModel"] = params["acc"]
            resultsOfTest["pred"] = pred[0]

            resultsOfTests.append(resultsOfTest)

        voteValue0 = 0
        voteValue1 = 0
        voted = 0

        for resTest in resultsOfTests:
            if resTest["accModel"] >= config["modelAffectionLimitator"]:
                if resTest["pred"] == 1:
                    voteValue1 = voteValue1 + resTest["accModel"]**config["accResultMultiplier"]
                else:
                    voteValue0 = voteValue0 + resTest["accModel"]**config["accResultMultiplier"]
  
        if voteValue1 > voteValue0:
            voted = 1
        else: 
            voted = 0

        if voted == y_var:
            gotRight += 1
        else:
            gotWrong += 1

    score = gotRight/(gotRight+gotWrong)
    overallscores.append(score)
    print(f"% {round(score,5)*100}")
    if score > BestScore and BestScore != 1 :
        BestScore = score

scTotal = 0
for sc in overallscores:
    scTotal+=sc

totalScores = scTotal = scTotal/len(overallscores)

print("Best Accuracy: %",round(BestScore,5)*100)
print("Average Accuracy: %",round(totalScores,5)*100)

warnings.resetwarnings()

# Time complexity = O(N*M) N is the size of the test data and M is the size of the training data. + Extra Loops