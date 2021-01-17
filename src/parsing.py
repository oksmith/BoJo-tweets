"""
Parse the JSON tweets data generated from `snscrape`. Save as CSV.
"""

import os
import json
import csv

import pandas as pd

from utils import get_date_from_file_loc


EXPECTED_COLUMNS = ['date', 'tweet_id', 'tweet_datetime', 'content', 'url', 'user',
       'user_id', 'user_verified', 'user_created', 'user_description',
       'user_location', 'user_followersCount', 'user_statusesCount',
       'outlinks', 'tcooutlinks', 'replyCount', 'retweetCount', 'likeCount',
       'quoteCount', 'lang', 'sourceLabel', 'retweetedTweet', 'quotedTweet',
       'n_mentionedUsers']


def read_json_tweets(json_file_loc):
    
    with open(json_file_loc, 'r') as f:
        lines = [line for line in f.read().split('\n') if len(line) > 0]
    
    return [json.loads(x) for x in lines]


def parse_json_tweets(tweets, date):

    return pd.DataFrame({
        'date': [date for _ in tweets],
        'tweet_id': [tweet['id'] for tweet in tweets],
        'tweet_datetime': [tweet['date'] for tweet in tweets],
        'content': [tweet['content'] for tweet in tweets],
        'url': [tweet['url'] for tweet in tweets],
        'user': [tweet['user']['username'] for tweet in tweets],
        'user_id': [tweet['user']['id'] for tweet in tweets],
        'user_verified': [tweet['user']['verified'] for tweet in tweets],
        'user_created': [tweet['user']['created'] for tweet in tweets],
        'user_description': [tweet['user']['description'] for tweet in tweets],
        'user_location': [tweet['user']['location'] for tweet in tweets],
        'user_followersCount': [tweet['user']['followersCount'] for tweet in tweets],
        'user_statusesCount': [tweet['user']['statusesCount'] for tweet in tweets],
        'outlinks': [tweet['outlinks'] for tweet in tweets],
        'tcooutlinks': [tweet['tcooutlinks'] for tweet in tweets],
        'replyCount': [tweet['replyCount'] for tweet in tweets],
        'retweetCount': [tweet['retweetCount'] for tweet in tweets],
        'likeCount': [tweet['likeCount'] for tweet in tweets],
        'quoteCount': [tweet['quoteCount'] for tweet in tweets],
        'lang': [tweet['lang'] for tweet in tweets],
        'sourceLabel': [tweet['sourceLabel'] for tweet in tweets],
        'retweetedTweet': [tweet['retweetedTweet'] for tweet in tweets],
        'quotedTweet': [tweet['quotedTweet'] for tweet in tweets],
        'n_mentionedUsers': [len(tweet['mentionedUsers']) if tweet['mentionedUsers'] is not None else 0 for tweet in tweets]
    })


def parse_directory(directory, output_csv_loc):
    
    df = pd.concat([
        parse_json_tweets(read_json_tweets(directory+'/'+file_loc), date=get_date_from_file_loc(file_loc))[EXPECTED_COLUMNS]
        for file_loc in os.listdir(directory) if file_loc.endswith('.json')
    ], axis=0)
    
    df = df.set_index('tweet_id')
    
    df.to_csv(output_csv_loc, quoting=csv.QUOTE_NONNUMERIC)
    print('Done!')
    
    
def parse_list_of_files(file_list, output_csv_loc):
    
    df = pd.concat([
        parse_json_tweets(read_json_tweets(file_loc), date=get_date_from_file_loc(file_loc))[EXPECTED_COLUMNS]
        for file_loc in file_list
    ])
    
    df = df.set_index('tweet_id')
    
    df.to_csv(output_csv_loc, quoting=csv.QUOTE_NONNUMERIC)
    print('Done!')
