youtube_api = "AIzaSyBeqjfz82NTL4aorMU-sC9A0pr52IoXRw0"

import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2
import os
import sys
from apiclient.discovery import build_from_document
from apiclient.errors import HttpError
# from oauth2client.client import flow_from_clientsecrets
# from oauth2client.file import Storage
# from oauth2client.tools import argparser, run_flow
import nltk
from random import shuffle
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from tensorflow.keras.models import Sequential
import streamlit as st
import tensorflow as tf

pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

DEVELOPER_KEY = youtube_api
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
CLIENT_SECRETS_FILE = "client_secrets.json"
YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"

# Calculate the start date as 1 year before today
today = date.today()
start_date = today - timedelta(days=5*365) 

# Format the start date as a string in the required format
START = start_date.strftime("%Y-%m-%d")
TODAY = today.strftime("%Y-%m-%d")

sia = SentimentIntensityAnalyzer()

unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

comments = []

def is_positive(comment: str) -> bool:
    """True if comment has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(comment)["compound"] > 0

def youtube_search(options, stock):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

    # Call the search.list method to retrieve results matching the specified
    # query term.
    search_response = youtube.search().list(
    q=str(stock) + ' stock performance',
    part='id,snippet',
    maxResults=options.max_results
    ).execute()

    videos = []
    videoIDs = []

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append('%s (%s)' % (search_result['snippet']['title'],
                                        search_result['id']['videoId']))
            videoIDs.append(search_result['id']['videoId'])

    max_comments = 200  # You can adjust this as needed

    for video in videoIDs:
        # Set an initial page token to start the pagination (None for the first page)
        page_token = None

        try:
            while len(comments) < max_comments:
                # Make a request to the API to get video comments
                comments_response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video,
                    textFormat='plainText',
                    maxResults=200,  # Maximum number of comments per page (you can adjust this)
                    pageToken=page_token
                ).execute()

                # Extract and append the comments
                for comment in comments_response['items']:
                    text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    # if text:
                    #     comments.append(comment)
                    comments.append(text)

                # Check if there are more pages of comments
                if 'nextPageToken' in comments_response:
                    page_token = comments_response['nextPageToken']
                else:
                    break  # No more pages

        except HttpError as e:
            print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

def predict_stock_prices(stock, num_future_points=30):
    # download the data
    df = yf.download(tickers=[stock], period='1y')
    y = df['Close'].ffill().bfill()
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)
    
    sentiment_scores = [sia.polarity_scores(comment)["compound"] for comment in comments]
    sentiment_scores = np.array(sentiment_scores)
    
    if len(X) != len(sentiment_scores):
        sentiment_scores = sentiment_scores[:len(X)]

    sentiment_scores = sentiment_scores.reshape(-1, 1)
    sentiment_scores = np.tile(sentiment_scores, (1,n_lookback))
    sentiment_scores = sentiment_scores.reshape(-1, n_lookback, 1) 

    combined_features = np.concatenate((X, sentiment_scores), axis=-1)
    

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 2)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(combined_features, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    # X_ = y[- n_lookback:]  # last available input sequence
    # print('shape of X_:', X_.shape)
    # X_ = X_.reshape(1, n_lookback, 2)

    # print("HELLO")

    # Y_ = model.predict(X_).reshape(-1, 1)
    # Y_ = scaler.inverse_transform(Y_)

    # Combine historical data (y) and sentiment scores
    y_reshaped = y[-n_lookback:].reshape(-1, 1)
    y_reshaped = np.tile(y_reshaped, (1,n_lookback))
    y_reshaped = y_reshaped.reshape(-1, n_lookback, 1) 

    print("Shape of y[-n_lookback:]:", y_reshaped.shape)
    print("Shape of sentiments[-n_lookback:]:", sentiment_scores[-n_lookback:].shape)

    combined_features_test = np.concatenate((y_reshaped, sentiment_scores[-n_lookback:]), axis=-1)

    # Reshape and make predictions
    # combined_features_test = combined_features_test.reshape(1, n_lookback, 2)
    # Y_ = model.predict(combined_features_test).reshape(-1, 1)
    # Y_ = scaler.inverse_transform(Y_)

    predictions = model.predict(combined_features_test)
    print("PREDICTIONS:", predictions)
    print("real values:", y)

    # organize the results in a data frame
    # df_past = df[['Close']].reset_index()
    # df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    # df_past['Date'] = pd.to_datetime(df_past['Date'])
    # df_past['Forecast'] = np.nan
    # df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    # df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    # df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    # df_future['Forecast'] = Y_.flatten()
    # df_future['Actual'] = np.nan

    # frames = [df_past, df_future]
    # results = pd.concat(frames).set_index('Date')

    # # plot the results
    # # results.plot(title='AAPL')

    # plt.plot(results)
    # plt.title('Prices Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # st.pyplot(plt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', help='Search term', default='Google')
    parser.add_argument('--max-results', help='Max results', default=25)
    args = parser.parse_args()

    st.title("Stock Price Prediction App")

    stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL): ")

    if st.button("Predict"):
        if stock_symbol:
            st.write(f"Predicting stock prices for {stock_symbol}...")

            try:
                youtube_search(args, stock_symbol)
                predict_stock_prices(stock_symbol)
            except (HttpError):
                print ('An HTTP error %d occurred:\n%s')
            except Exception as e:
                st.error(f"An error occurred: {e}")