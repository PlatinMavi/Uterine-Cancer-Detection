import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
import pickle
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore', category=UserWarning)

path = os.getcwd()+r"\Datasets\default.csv"
df = pd.read_csv(path)
df.replace("-", "0.0", inplace=True)
df.replace("----", "0.0", inplace=True)

def ConvertFloat(value):
    return float(value.replace(',', '.'))

for column in df.columns[1:]:
    df[column] = df[column].apply(ConvertFloat)

# Define the target column mapping
status = {"basitler": 0, "ein": 0, "highrisk": 1, "lowrisk": 1}
yUnmapped = df['status']
y = yUnmapped.map(status)
X = df.drop(columns=['status',"NRBC"])

BestScore = 0
BestScoreSeed = 0
overAllScoreStorage = []

for rs in range(10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) #random_state=rs

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
    ]

    all_columns = X.columns.tolist()
    columns = all_columns[:]

    for col in columns:
        if "_" in col:
            columns.pop(columns.index(col))

    for cosCol in columns:
        TestParameters.append({"params":[cosCol], "name":cosCol})

    knn_classifier = KNeighborsClassifier(n_neighbors=20)

    def TrainKnn(selected_features):
        selected_columns = list(selected_features)
        knn_classifier.fit(X_train[selected_columns], y_train)
        y_pred = knn_classifier.predict(X_machine[selected_columns])
        accuracy = accuracy_score(y_machine, y_pred)
        return accuracy

    def TestKnn(selected_features, x_var):
        selected_columns = list(selected_features)
        knn_classifier.fit(X_train[selected_columns], y_train)
        # Reshape x_var to a 2D array
        x_var_2d = x_var[selected_columns].values.reshape(1, -1)
        y_pred = knn_classifier.predict(x_var_2d)
        
        return y_pred
    
    #IN PROGRESS FOR ENSEMBLE LEARNING

    random_forest_classifier = RandomForestClassifier()

    def TrainRandomForest(selected_features):
        selected_columns = list(selected_features)
        random_forest_classifier.fit(X_train[selected_columns], y_train)
        y_pred = random_forest_classifier.predict(X_machine[selected_columns])
        accuracy = accuracy_score(y_machine, y_pred)
        return accuracy

    def TestRandomForest(selected_features, x_var):
        selected_columns = list(selected_features)
        random_forest_classifier.fit(X_train[selected_columns], y_train)
        x_var_2d = x_var[selected_columns].values.reshape(1, -1)
        y_pred = random_forest_classifier.predict(x_var_2d)
        
        return y_pred

    for parameters in TestParameters:
        acc1 = TrainKnn(parameters["params"])
        parameters["acc1"] = acc1*10

        acc2 = TrainRandomForest(parameters["params"])
        parameters["acc2"] = acc2*10
        # print(acc, parameters["name"])

    gotRight = 0
    gotWrong = 0

    for i,useless in enumerate(X_hand):
        x_var = X_hand.reset_index(drop=True).iloc[i]
        y_var = y_hand.reset_index(drop=True).iloc[i]

        resultsOfTests = []
        for params in TestParameters:
            resultsOfTest = {}
            pred = TestKnn(params["params"],x_var)
            resultsOfTest["accModel1"] = params["acc1"]
            resultsOfTest["pred1"] = pred[0]

            pred = TestRandomForest(params["params"],x_var)
            resultsOfTest["accModel2"] = params["acc2"]
            resultsOfTest["pred2"] = pred[0]

            resultsOfTests.append(resultsOfTest)

        voteValue0 = 0
        voteValue1 = 0
        voted = 0

        for resTest in resultsOfTests:
            # if resTest["accModel"] >= 0.60:
                if resTest["pred1"] == 1:
                    voteValue1 = voteValue1 + resTest["accModel1"]**3
                else:
                    voteValue0 = voteValue0 + resTest["accModel1"]**3

                if resTest["pred2"] == 1:
                    voteValue1 = voteValue1 + resTest["accModel2"]**2
                else:
                    voteValue0 = voteValue0 + resTest["accModel2"]**2
        
        if voteValue1 > voteValue0:
            voted = 1
        else: 
            voted = 0

        if voted == y_var:
            gotRight += 1
        else:
            gotWrong += 1

    score = gotRight/(gotRight+gotWrong)
    print(score)
    overAllScoreStorage.append(score)
    # if score > BestScore and BestScore != 1 :
    #     BestScore = score
    #     BestScoreSeed = rs
    #     print(BestScore)
        
    #     with open('knn_model.pkl', 'wb') as model_file:
    #         pickle.dump(knn_classifier, model_file)
    #     print(f"New best score ({BestScore}) achieved and the model is saved.")

# print(BestScore,BestScoreSeed)

scTotal = 0
for sc in overAllScoreStorage:
    scTotal = scTotal + sc
print("Overall Score:",scTotal/len(overAllScoreStorage)*100)

warnings.resetwarnings()