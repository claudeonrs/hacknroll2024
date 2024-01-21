from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from sentiment_analysis import roberta_classification
from scraper import get_ids_shopee, get_reviews_shopee
from emotion_analysis import emoroberta_classification
import pandas as pd

app = Flask(__name__)

emotions_dict_emojis = {
    'disappointment': '😞',
    'sadness': '😢',
    'annoyance': '😠',
    'neutral': '😐',
    'disapproval': '👎',
    'realization': '😲',
    'nervousness': '😬',
    'approval': '👍',
    'joy': '😄',
    'anger': '😡',
    'embarrassment': '😳',
    'caring': '❤️',
    'remorse': '😔',
    'disgust': '🤢',
    'grief': '😥',
    'confusion': '😕',
    'relief': '😌',
    'desire': '🔥',
    'admiration': '😍',
    'optimism': '😊',
    'fear': '😨',
    'love': '💖',
    'excitement': '😃',
    'curiosity': '🤔',
    'amusement': '😆',
    'surprise': '😯',
    'gratitude': '🙏',
    'pride': '🌟'}



@app.route('/')
def display_input_page():
    session['emotions_dict_emojis'] = emotions_dict_emojis
    return render_template('base.html')

@app.route('/api/get_sentiments', methods=['GET'])
def get_sentiments():
    url = request.args.get('url', 'None')
    limit = request.args.get('limit', 10)

    shopid, itemid = get_ids_shopee(url)
    rating_count, df = get_reviews_shopee(itemid, shopid, limit=limit)
    df['comment'] = df['comment'].str.replace('\n', ' ')

    roberta_class = df['comment'].map(roberta_classification)
    roberta_class_dict = roberta_class.map(lambda d: max(d, key=d.get)).value_counts().to_dict()

    emoroberta_class = df['comment'].map(emoroberta_classification)
    emoroberta_class_dict = emoroberta_class.value_counts().to_dict()

    # comment_data
    # process df['comment']
    
    # data = {'sentiment_data': roberta_class_dict,
    #         'emotions_data': emoroberta_class_dict}
    session['sentiment_data'] = roberta_class_dict
    session['emotions_data'] = emoroberta_class_dict
    session['emotions_dict_emojis'] = emotions_dict_emojis
    
    # return render_template('result.html', sentiment_data=session['sentiment_data'], emotions_data=session['emotions_data'])
    return redirect(url_for('get_result'))

@app.route('/result', methods=['GET'])
def get_result():
    sentiment_data = session['sentiment_data']
    emotions_data = session['emotions_data']
    # return render_template('result.html')
    return render_template('result.html', sentiment_data=sentiment_data, emotions_data=emotions_data, emotions_dict_emojis=emotions_dict_emojis)

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True)