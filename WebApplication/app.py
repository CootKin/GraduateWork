from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json


app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(host="192.168.0.101", port="5000", debug=False)
