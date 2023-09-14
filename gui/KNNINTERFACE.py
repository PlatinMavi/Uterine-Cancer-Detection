import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import warnings

#STOP ANNOYING WARNINGS
warnings.filterwarnings('ignore', category=UserWarning)

def Start(values):

    # FIXING DATASET
    path = os.getcwd()+r"\Datasets\default.csv"
    df = pd.read_csv(path)
    df.replace("-", "0.0", inplace=True)
    df.replace("----", "0.0", inplace=True)

    def ConvertFloat(value):
        return float(value.replace(',', '.'))

    for column in df.columns[1:]:
        df[column] = df[column].apply(ConvertFloat)

    # TARGETING COLUMNS AND MAPPING FOR EASIER PREDICTIONS
    status = {"basitler": 0, "ein": 0, "highrisk": 1, "lowrisk": 1}
    yUnmapped = df['status']
    y = yUnmapped.map(status)
    X = df.drop(columns=['status',"NRBC"]) # LACK OF DATA FOR NRBC

    # SPLITING FOR DIFFERENT TEST VARIABLES 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=86)
    X_hand, X_machine, y_hand, y_machine = X_test.iloc[:len(X_test)//2], X_test.iloc[len(X_test)//2:], y_test.iloc[:len(y_test)//2], y_test.iloc[len(y_test)//2:]


    # COLUMNS AND COMBINATIONS THAT HAS HIGH ACCURACY RATE
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

    # TESTING INDIVIDUAL VALUES AND ADDING THEM AS TEST PARAMETERS
    all_columns = X.columns.tolist()
    columns = all_columns[:]

    for col in columns:
        if "_" in col:
            columns.pop(columns.index(col))

    for cosCol in columns:
        TestParameters.append({"params":[cosCol], "name":cosCol})

    # KNN ALORITHIM DEFINITION
    knn_classifier = KNeighborsClassifier(n_neighbors=20)

    # TO GET ACCURACY SCORE FOR INDIVIDUAL TESTING
    def TrainKnn(selected_features):
        selected_columns = list(selected_features)
        knn_classifier.fit(X_train[selected_columns], y_train)
        y_pred = knn_classifier.predict(X_machine[selected_columns])
        accuracy = accuracy_score(y_machine, y_pred)
        return accuracy

    # TO GET PREDICTIONS FOR INDIVIDUALS
    def TestKnn(selected_features, x_var):
        selected_columns = list(selected_features)
        knn_classifier.fit(X_train[selected_columns], y_train)
        # Reshape x_var to a 2D array
        x_var_2d = x_var[selected_columns].values.reshape(1, -1)
        y_pred = knn_classifier.predict(x_var_2d)
        
        return y_pred
    
    def TestKnnINTERFACE(selected_features, x_var):
        selected_columns = list(selected_features)
        knn_classifier.fit(X_train[selected_columns], y_train)
        # Reshape x_var to a 2D array
        x_var_2d = x_var[selected_columns].values.reshape(1, -1)
        y_pred = knn_classifier.predict(x_var_2d)
        
        return y_pred

    # SAVING ACCURACY RATE FOR LATER USEAGES
    for parameters in TestParameters:
        acc = TrainKnn(parameters["params"])
        parameters["acc"] = acc*10


    gotRight = 0
    gotWrong = 0

    # TESTING INDIVIDUAL DATA IN EVERY COMB AND CREATED VOTING SYSTEM
    for i,useless in enumerate(X_hand):
        x_var = X_hand.reset_index(drop=True).iloc[i]
        y_var = y_hand.reset_index(drop=True).iloc[i]

        resultsOfTests = []
        for params in TestParameters:
            resultsOfTest = {}
            pred = TestKnn(params["params"],x_var)
            resultsOfTest["accModel"] = params["acc"]
            resultsOfTest["pred"] = pred[0]

            resultsOfTests.append(resultsOfTest)

        voteValue0 = 0
        voteValue1 = 0
        voted = 0

        for resTest in resultsOfTests:
            # if resTest["accModel"] < 0.67:
                if resTest["pred"] == 1:
                    voteValue1 = voteValue1 + resTest["accModel"]**5
                else:
                    voteValue0 = voteValue0 + resTest["accModel"]**5
        
        if voteValue1 > voteValue0:
            voted = 1
        else: 
            voted = 0

        if voted == y_var:
            gotRight += 1
        else:
            gotWrong += 1

    score = gotRight/(gotRight+gotWrong) # CALCULATE OVERALL ACCURACY

    resultsOfTestsd = []
    x_vard = values
    for paramsd in TestParameters:
        resultsOfTestd = {}
        predd = TestKnnINTERFACE(params["params"],x_vard,)
        resultsOfTestd["accModel"] = paramsd["acc"]
        resultsOfTestd["pred"] = predd[0]

        resultsOfTestsd.append(resultsOfTestd)

    voteValue0d = 0
    voteValue1d = 0
    votedd = 0

    for resTestd in resultsOfTests:
        # if resTest["accModel"] < 0.67:
            if resTestd["pred"] == 1:
                voteValue1d = voteValue1d + resTestd["accModel"]**2
            else:
                voteValue0d = voteValue0d + resTestd["accModel"]**2
    
    if voteValue1d > voteValue0d:
        votedd = 1
    else: 
        votedd = 0

    return score, votedd

s = Start([5.16, 2.910, 1.700, 0.380 ,7.4, 2.9, 0.4, 9.80, 29.5, 87.30, 29.00, 33.20, 43.10, 13.60, 256, 10.90, 0.28, 12.20])


warnings.resetwarnings()