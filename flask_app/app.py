from flask import Flask, render_template, request
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# YouTube API key
API_KEY = 'AIzaSyCV3uvAkRrRieAa5nYPnsuAijy3mOS7kkc'

# Initialize the Vader sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to fetch YouTube comments
def get_youtube_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    response = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100).execute()
    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments

# Function to categorize sentiment using Vader
def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function for sentiment analysis using Vader
def analyze_sentiment_vader(text):
    # Analyze sentiment
    vader_scores = vader_analyzer.polarity_scores(text)
    compound_score = vader_scores['compound']
    sentiment_label = categorize_sentiment(compound_score)
    return sentiment_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form['video_url']
    video_id = video_url.split('=')[-1]
    comments = get_youtube_comments(video_id)
    sentiments = [analyze_sentiment_vader(comment) for comment in comments]
    # Zip comments and sentiments before passing them to the template
    comment_sentiment_pairs = list(zip(comments, sentiments))
    return render_template('results.html', comment_sentiment_pairs=comment_sentiment_pairs)

if __name__ == '__main__':
    app.run(debug=True)
