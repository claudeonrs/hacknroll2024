from flask import Blueprint, render_template

auth = Blueprint('auth', __name__)

@auth.route('/result')
def login():
    return render_template("result.html")