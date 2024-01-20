from flask import Blueprint

auth = Blueprint('auth', __name__)

@auth.route('/result')
def login():
    return "<p>Result</p>"