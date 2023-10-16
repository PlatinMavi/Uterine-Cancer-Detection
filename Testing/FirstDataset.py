import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

path = os.getcwd()+r"\Datasets\default.csv"
path = r"C:\Users\PC\Desktop\Cervicular-Cancer-Detection\Datasets\default.csv"

# Load the dataset
df = pd.read_csv(path)

# Preprocessing steps
df.replace("-", "0.0", inplace=True)
df.replace("----", "0.0", inplace=True)

def ConvertFloat(value):
    return float(value.replace(",", "."))

for column in df.columns[1:]:
    df[column] = df[column].apply(ConvertFloat)

# Define the target column mapping
status = {"basitler": 0, "ein": 0, "highrisk": 1, "lowrisk": 1}
yUnmapped = df["status"]
y = yUnmapped.map(status)

# Split data into training and testing sets
X = df.drop(columns=["status"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=86)

# Create a KNeighborsClassifier instance
k = 20
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Function to calculate accuracy for a given feature subset
def get_accuracy(selected_features):
    selected_columns = list(selected_features)
    knn_classifier.fit(X_train[selected_columns], y_train)
    y_pred = knn_classifier.predict(X_test[selected_columns])
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Get a list of all column names (excluding "status")
all_columns = X.columns.tolist()
columns = all_columns[:]

for col in columns:
    if "_" in col:
        columns.pop(columns.index(col))

test1 = ["MCHC", "RDWSD", "RDWCV", "PCT", "PDW"]
test2 = ["WBC", "MCV", "MCHC", "RDWSD", "RDWCV", "PCT", "PDW"]
test3 = ["EOS", "BASO", "WBC", "MONO", "HCT", "MCHC", "RDWSD", "RDWCV", "MPV", "PCT", "PDW"]
results = [test1,test2, test3]


print("\n COMBINATUAL SCORE \n")

for i,res in enumerate(results):
    print("\033[91m"+str(i), "\033[92m"+str(get_accuracy(res)*100),"\033[94m")

print("\n INDIVIDUAL SCORE \n")


best = []

for i,rep in enumerate(columns):
    score = get_accuracy([rep])*100
    print("\033[91m"+str(i), "\033[92m"+str(score),"\033[94m"+str(rep))
    if score >= 67:
        best.append(rep)

# print(best)
list("LYM", "BASO", "HGB", "HCT", "MCV", "MCHC", "RDWSD", "RDWCV", "PCT", "PDW")

# ["WBC", "NEU", "NEU_P", "LYM", "LYM_P", "MONO", "MONO_P", "EOS", "EOS_P", "BASO", "BASO_P", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDWSD", "RDWCV", "PLT", "MPV", "PCT", "PDW", "NRBC", "NRBC_P"]
# ["WBC", "NEU", "LYM", "MONO", "EOS", "BASO", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDWSD", "RDWCV", "PLT", "MPV", "PCT", "PDW", "NRBC"]