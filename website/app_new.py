from flask import Flask, jsonify, request, render_template
from sentiment_analysis import roberta_classification
from scraper import get_ids_shopee, get_reviews_shopee
from emotion_analysis import emoroberta_classification
from comment_analysis import bart_classification
import pandas as pd

app = Flask(__name__)

@app.route('/')
def display_input_page():
    return render_template('base.html')

@app.route('/api/get_sentiments', methods=['GET'])
def get_sentiments():
    url = request.args.get('url', 'None')
    limit = request.args.get('limit', 10)

    shopid, itemid = get_ids_shopee(url)
    rating_count, df = get_reviews_shopee(itemid, shopid, limit=limit)
    df['comment'] = df['comment'].str.translate(str.maketrans('\n\t', ' '))

    roberta_class = df['comment'].map(roberta_classification)
    roberta_class_dict = roberta_class.map(lambda d: max(d, key=d.get)).value_counts().to_dict()

    emoroberta_class = df['comment'].map(emoroberta_classification)
    emoroberta_class_dict = emoroberta_class.value_counts().to_dict()

    data = {'sentiment_data': roberta_class_dict,
            'emotions_data': emoroberta_class_dict,
            'comments_data':_list}

    return jsonify(data)
@app.route('/api/get_comments', methods=['GET'])
def get_comment_summaries():
    url = request.args.get('url', 'None')
    limit = request.args.get('limit', 10)

    shopid, itemid = get_ids_shopee(url)
    rating_count, df = get_reviews_shopee(itemid, shopid, limit=limit)
    df['comment'] = df['comment'].str.translate(str.maketrans('\n\t', ' '))
    data = []

    STAR_MAX_COUNT,comment_agg=5,{}
    for star_count in range(1,STAR_MAX_COUNT+1):
        comment_agg[star_count]=list(df[df['rating_star']==star_count]['comment'])
        data.append(bart_classification(comment_agg[star_count]))
    
    return jsonify(data)
if __name__ == '__main__':
    app.run(debug=True)