import requests
import pandas as pd
import re

def get_ids_shopee(prod_url):
    shopid, itemid = re.findall('i\.(\d+)\.(\d+)\?', prod_url)[0]
    return shopid, itemid

def get_reviews_shopee(itemid, shopid, limit=None, limit_per_req = 59, offset=0): 
    '''
    Get reviews from Shopee website in terms of data frame

    get itemid and shopid from the get_ids_shopee function
    '''
    url = "https://shopee.sg/api/v2/item/get_ratings"

    querystring = {
        "exclude_filter":"1",
        "filter":"1", #! 1 only includes those with comments
                      #! 0 includes all with/without comments
        "filter_size":"0",
        "flag":"1",
        "fold_filter":"0",
        "itemid":itemid,
        "limit":str(limit_per_req),
        "offset":str(offset),
        "relevant_reviews":"false",
        "request_source":"2",
        "shopid":shopid,
        "tag_filter":"",
        "type":"0",
        "variation_filters":""}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    rating_total = response.json()['data']['item_rating_summary']['rating_total']
    rating_count = response.json()['data']['item_rating_summary']['rating_count']
    rcount_with_context = response.json()['data']['item_rating_summary']['rcount_with_context']


    rating_list = list()
    if limit is None:
        limit = rcount_with_context

    pages_to_scrape = limit//limit_per_req + 1
    for i in range(pages_to_scrape):
        querystring['offset'] = str(offset)
        response = requests.request("GET", url, headers=headers, params=querystring)
        if response.json()['data']['ratings'] is not None:
            rating_list += response.json()['data']['ratings']
        offset += limit_per_req
    
    rating_df = pd.json_normalize(rating_list)[:limit]
    return rating_count, rating_df