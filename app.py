import streamlit as st
import requests  
import pandas as pd  
import matplotlib.pyplot as plt  
from textblob import TextBlob  
from wordcloud import WordCloud  
from alpha_vantage.timeseries import TimeSeries  
import nltk  
nltk.download("punkt")

# API Keys (Replace with your keys)
NEWS_API_KEY = "your_newsapi_key_here"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"
STOCK_SYMBOL = "TSLA"

# Fetch Stock News from NewsAPI
def fetch_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        return []

# Perform Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def categorize_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Fetch Real-time Stock Price from Alpha Vantage
def get_stock_price(symbol):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    data, meta_data = ts.get_intraday(symbol=symbol, interval="5min", outputsize="compact")
    latest_price = data.iloc[0]["4. close"]
    return latest_price

# Streamlit UI
st.title("📈 Stock Sentiment Analysis")
st.sidebar.header("Settings")
company = st.sidebar.text_input("Enter Company Name", STOCK_SYMBOL)

if st.sidebar.button("Analyze"):
    news_data = fetch_news(company)
    if news_data:
        headlines = [article["title"] for article in news_data]
        df = pd.DataFrame(headlines, columns=["Headline"])
        df["Sentiment Score"] = df["Headline"].apply(get_sentiment)
        df["Sentiment Category"] = df["Sentiment Score"].apply(categorize_sentiment)

        # Display Results
        st.subheader(f"📊 Sentiment Analysis for {company}")
        st.dataframe(df[["Headline", "Sentiment Category"]])

        # Stock Price
        current_price = get_stock_price(STOCK_SYMBOL)
        st.subheader(f"💰 Current {STOCK_SYMBOL} Price: ${current_price}")

        # Histogram
        st.subheader("📊 Sentiment Distribution")
        plt.figure(figsize=(10, 5))
        plt.hist(df["Sentiment Score"], bins=10, color="skyblue", edgecolor="black")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Number of Headlines")
        plt.title(f"Sentiment Distribution for {company} News")
        st.pyplot(plt)

        # WordCloud
        st.subheader("🔠 Most Frequent Words")
        text = " ".join(df["Headline"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    else:
        st.warning("⚠️ No news articles found! Try another company.")

