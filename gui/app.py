from flask import Flask, render_template,redirect, url_for, request, send_file
import KNNINTERFACE

app = Flask(__name__)
app.config["SECRET_KEY"] = "key"
app.config["UPLOAD_FOLDER"] = "static/files"

@app.route("/", methods=["GET", "POST"])
def GetImage():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)