import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Bearer token (replace 'your_bearer_token' with your actual bearer token)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAALPhsgEAAAAAKLfrxUc3jxpcMl40VB9AFFSHh3I%3DtZw7LdZgb4spegShZ9z1Gz6A8Dpy2jjY07qPlwZWFMUH7DQrAo'
consumer_key = '1212685649825783808-ltSlzQmyKz9VXRdG3yw29Wjq3hsBYh'
consumer_secret = 'oz9hP6PVAQZxcOJ9DHJL1S6vNlvOThAh692XjlHKypYOv'

# Authenticate with Twitter API using bearer token
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
auth.apply_auth()
api = tweepy.API(auth, wait_on_rate_limit=True)

# Check if authentication was successful
if not api:
    print("Authentication failed. Please check your bearer token.")
    exit()

# Get tweets using Twitter API
user_name = "@premierleague"  #the Twitter username you want to fetch tweets from
tweet_count = 10  # Number of tweets to fetch
tweets = []
for tweet in tweepy.Cursor(api.user_timeline, screen_name=user_name, tweet_mode="extended").items(tweet_count):
    tweets.append(tweet.full_text)

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# Sentiment analysis for each tweet
for i, tweet in enumerate(tweets, start=1):
    print(f"Tweet {i}: {tweet}")

    # Sentiment analysis
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    for j in range(len(scores)):
        label = labels[j]
        score = scores[j]
        print(f'Sentiment: {label}, Score: {score}')

    print("-" * 50)
