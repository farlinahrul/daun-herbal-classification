from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('template_header.html')

@app.route("/ann")
def ann():
    return render_template('ann.html')

@app.route("/cnn")
def cnn():
    return render_template('cnn.html')