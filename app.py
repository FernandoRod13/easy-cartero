from flask import Flask
import os
import socket
from model import *


app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>" \
           "<b>Visits:</b> {visits}"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname(), visits=0)


@app.route("/train")
def train_model():
    Model().train_model()

@app.route("/predict")
def predict():
    Model().predict()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
