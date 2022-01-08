import sklearn
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from joblib import load
import tweepy
import time
from flask import Flask, render_template, request, redirect, url_for
from joblib import load

# -----------------------------------------------------------------------------------

pd.set_option('display.max_colwidth', 1000)

api_key = "QsVk97gXC6Y0ISWIk3Mn1jl7q"
api_secret_key = "mkna7mzVmboExLIG3azuyWI6NwjaUeyuIS15C1OkuEYNoagxWh"
access_token = "361125027-KS9CHs4dgsB5sfLdaBJWV53R6QulPRwSrqn5cVrO"
access_token_secret = "RnGLDhZBaxqtBkQMx9R30WUedkIHQW2OC1rVWl1j2HNtz"

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def get_related_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 50
    try:
        # Pulling individual tweets from query
        for tweet in api.search(q=text_query, count=count):
            print(tweet.text)
            # Adding to list that contains all tweets
            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_text': tweet.text})
        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)

# --------------------------------------------------------------------------------


# load the pipeline object
pipeline = load("text_classification.joblib")

# function to get results for a particular text query
def requestResults(name):
    # get the tweets text
    tweets = get_related_tweets(name)
    # get the prediction
    tweets['prediction'] = pipeline.predict(tweets['tweet_text'])
    # get the value counts of different labels predicted
    data = str(tweets.prediction.value_counts()) + '\n\n'
    return data + str(tweets)

# ------------------------------------------------------------------------------

app = Flask(__name__)

# render default webpage
@app.route('/')
def home():
    return render_template('sentimentanalysis.html')

# when the post method detect, then redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

# get the data for the requested query
@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "

app.run(debug=True)