from flask import Flask, render_template, jsonify, request
from KNNINTERFACE import Predict

app = Flask(__name__)
app.config["SECRET_KEY"] = "key"
app.config["UPLOAD_FOLDER"] = "static/files"

@app.route("/", methods=["GET", "POST"])
def GetImage():
    return render_template("index.html")

@app.route("/api/prediction", methods=["POST"])
def ReturnApi():
    data = request.json
    dataToPass = [float(value) for value in list(data.values())]
    prediciton,v0,v1 = Predict(dataToPass)

    return jsonify({"p":prediciton,"v0":v0,"v1":v1})

if __name__ == '__main__':
    app.run(debug=True)