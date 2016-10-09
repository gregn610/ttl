from flask import Flask
from flask_run import flask_run

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

flask_run(app)
