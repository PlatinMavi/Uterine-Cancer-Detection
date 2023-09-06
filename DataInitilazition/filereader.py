import PyPDF2
import csv
import os

headerWprevious = ["status","WBC","NEU","NEU_P","LYM","LYM_P","MONO","MONO_P","EOS","EOS_P","BASO","BASO_P","RBC","HGB","HCT","MCV","MCH","MCHC","RDWSD","RDWCV","PLT","MPV","PCT","PDW","NRBC","NRBC_P",
    "1WBC","1NEU","1NEU_P","1LYM","1LYM_P","1MONO","1MONO_P","1EOS","1EOS_P","1BASO","1BASO_P","1RBC","1HGB","1HCT","1MCV","1MCH","1MCHC","1RDWSD","1RDWCV","1PLT","1MPV","1PCT","1PDW","1NRBC","1NRBC_P",
    "2WBC","2NEU","2NEU_P","2LYM","2LYM_P","2MONO","2MONO_P","2EOS","2EOS_P","2BASO","2BASO_P","2RBC","2HGB","2HCT","2MCV","2MCH","2MCHC","2RDWSD","2RDWCV","2PLT","2MPV","2PCT","2PDW","2NRBC","2NRBC_P",
    "3WBC","3NEU","3NEU_P","3LYM","3LYM_P","3MONO","3MONO_P","3EOS","3EOS_P","3BASO","3BASO_P","3RBC","3HGB","3HCT","3MCV","3MCH","3MCHC","3RDWSD","3RDWCV","3PLT","3MPV","3PCT","3PDW","3NRBC","3NRBC_P",]

headerDefault = ["status","WBC","NEU","NEU_P","LYM","LYM_P","MONO","MONO_P","EOS","EOS_P","BASO","BASO_P","RBC","HGB","HCT","MCV","MCH","MCHC","RDWSD","RDWCV","PLT","MPV","PCT","PDW","NRBC","NRBC_P"]

with open("default.csv", "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headerDefault)

path = "C:/Users/PC/Desktop/data/"

def GET_GROUPS(path):
    dir = os.listdir(path)

    files = {}

    for group in dir:
        npath = path+group
        ndir = os.listdir(npath)
        files[group] = ndir

    return files  

def GET_PARAMS_AND_WRITE(path,status):
    

    try:
        reader = []
        FPDF = open(path,"rb")

        txt = PyPDF2.PdfReader(FPDF)._get_page(0).extract_text().split("Hemogram")[1].split("Umuttepe")[0].split("      ")
        for j in txt:
            hgh = j.split("\n")
            for jhj in hgh:
                reader.append(jhj)

        WBC = reader[reader.index("WBC (Lökosit) ")+1].split(" ")
        NEU = reader[reader.index("NEU (Nötrofil Sayısı) ")+1].split(" ")
        NEU_P = reader[reader.index("NEU % (Nötrofil Yüzdesi) ")+1].split(" ")
        LYM = reader[reader.index("LYM (Lenfosit Sayısı) ")+1].split(" ")
        LYM_P = reader[reader.index("LYM % (Lenfosit Yüzdesi) ")+1].split(" ")
        MONO = reader[reader.index("MONO (Monosit Sayısı) ")+1].split(" ")
        MONO_P = reader[reader.index("MONO % (Monosit Yüzdesi) ")+1].split(" ")
        EOS = reader[reader.index("EOS (Eozinofil Sayısı) ")+1].split(" ")
        EOS_P = reader[reader.index("EOS % (Eozinofil Yüzdesi) ")+1].split(" ")
        BASO = reader[reader.index("BASO (Basofil Sayısı) ")+1].split(" ")
        BASO_P = reader[reader.index("BASO % (Basofil Yüzdesi) ")+1].split(" ")
        RBC = reader[reader.index("RBC (Eritrosit) ")+1].split(" ")
        HGB = reader[reader.index("HGB (Hemoglobin) ")+1].split(" ")
        HCT = reader[reader.index("HCT (Hematokrit) ")+1].split(" ")
        MCV = reader[reader.index("MCV (Ortalama Eritrosit Hacmi) ")+1].split(" ")
        MCH = reader[reader.index("MCH (Ortalama Hücre Hemoglobin) ")+1].split(" ")
        MCHC = reader[reader.index("MCHC (Ortalama Hücre Hemog.Konsant.) ")+1].split(" ")
        PLT = reader[reader.index("PLT (Trombosit) ")+1].split(" ")
        MPV = reader[reader.index("MPV (Ortalama Trombosit Hacmi) ")+1].split(" ")
        try:
            RDWSD = reader[reader.index("RDW-SD ")+1].split(" ")
        except:
            RDWSD = "-"

        try:
            RDWCV = reader[reader.index("RDW-CV ")+1].split(" ")

        except:
            RDWCV = "-"
        try:
            PCT = reader[reader.index("PCT (Platekrit) ")+1].split(" ")

        except:
            PCT = "-"

        try:
            PDW = reader[reader.index("PDW (Trombosit Dağılım Genişliği) ")+1].split(" ")
        except:
            PDW = "-"

        try:
            NRBC = reader[reader.index("NRBC ")+1].split(" ")
        except:
            NRBC = "-"

        try:
            NRBC_P = reader[reader.index("NRBC % ")+1].split(" ")
        except:
            NRBC_P = "-"
        
        FPDF.close()

        with open("default.csv", "a", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            data = [status,WBC[-1],NEU[-1],NEU_P[-1],LYM[-1],LYM_P[-1],MONO[-1],MONO_P[-1],EOS[-1],EOS_P[-1],BASO[-1],BASO_P[-1],RBC[-1],HGB[-1],HCT[-1],MCV[-1],MCH[-1],MCHC[-1],RDWSD[-1],RDWCV[-1],PLT[-1],MPV[-1],PCT[-1],PDW[-1],NRBC[-1],NRBC_P[-1]]
            writer.writerow(data)
        
        return print("complete")
    except Exception as e:
        print("hata",e)

fl = GET_GROUPS(path)

for group in fl:
    for fls in fl[group]:
        GET_PARAMS_AND_WRITE(path+group+"/"+fls,group)

# reader = []
# FPDF = open(path+"lowrisk/ZULEYHA_GIRIK_20230117_2807997_LabRapor.pdf","rb")

# txt = PyPDF2.PdfReader(FPDF)._get_page(0).extract_text().split("Hemogram")[1].split("Umuttepe")[0].split("      ")
# for j in txt:
#     hgh = j.split("\n")
#     for jhj in hgh:
#         reader.append(jhj)

# print(reader)