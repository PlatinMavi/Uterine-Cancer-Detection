import pandas as pd
from sklearn.model_selection import train_test_split
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

warnings.filterwarnings('ignore', category=UserWarning)

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
    
    def TrainKnn(self, selected_features, X_train, X_test, y_train, y_test):
        selected_columns = list(selected_features)
        self.fit(X_train[selected_columns], y_train)
        y_pred = []
        for i in range(len(X_test)):
            x_var = X_test[selected_columns].iloc[i]
            y_pred.append(self.TestKnn(selected_features, X_train, y_train, x_var))  # Provide all required arguments
        accuracy = self.CalculateAccuracy(y_test, y_pred)
        return accuracy

    def TestKnn(self, selected_features, X_train, y_train, x_var):
        selected_columns = list(selected_features)
        self.fit(X_train[selected_columns], y_train)
        try:
            x_var_2d = x_var[selected_columns].values.reshape(1, -1)
        except:
            newValues = [[]]
            allparams = ['WBC', 'NEU', 'LYM', 'MONO', 'EOS', 'BASO', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDWSD', 'RDWCV', 'PLT', 'MPV', 'PCT', 'PDW', 'NRBC']

            for param in selected_columns:
                newValues[0].append(x_var[allparams.index(param)])

            x_var_2d = newValues
        # print(x_var_2d)
        y_pred = self.predict(x_var_2d)
        return y_pred


def Predict(xFromPost):
    path = r"C:\Users\PC\Desktop\Cervicular-Cancer-Detection\gui\default.csv"
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
    X = df.drop(columns=['status'])

    KnnAlgorithm = Knn(config["n_neighbor"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config["test_size"],)

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
        }
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
        acc = KnnAlgorithm.TrainKnn(parameters["params"],X_train,X_test,y_train,y_test)
        parameters["acc"] = acc*config["accMultiplier"]

    x_var = xFromPost
    resultsOfTests = []
    for params in TestParameters:
        resultsOfTest = {}
        pred = KnnAlgorithm.TestKnn(params["params"], X_train, y_train, x_var)  # Provide all required arguments
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

    return voted,voteValue0,voteValue1

warnings.resetwarnings()