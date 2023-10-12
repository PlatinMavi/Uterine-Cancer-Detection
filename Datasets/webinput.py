import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
import pickle
import sqlite3
data = []
vt = sqlite3.connect("blood.db")

im = vt.cursor()
while True:
    input_data = input("Verileri virgülle ayirarak girin (Çikmak için 'exit' yazin): ")
    if input_data.lower() == "exit":
        break
    try:
        data_row = [float(x.strip()) for x in input_data.split(',')]
        data.append(data_row)
        im.execute("INSERT INTO blood VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", data_row)
    except ValueError:
        print("Geçersiz giriş. Lütfen tekrar deneyin veya 'exit' yazarak çikin.")
    

user_df = pd.DataFrame(data, columns=["WBC","NEU","NEU_P","LYM","LYM_P","MONO","MONO_P","EOS","EOS_P","BASO","BASO_P","RBC","HGB","HCT","MCV","MCH","MCHC","RDWSD","RDWCV","PLT","MPV","PCT","PDW","NRBC","NRBC_P"])



vt.commit()
vt.close()



