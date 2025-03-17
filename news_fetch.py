import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download("punkt")
nltk.download("stopwords")

API_KEY = "Replace with your News_API key"  
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_news(query):
    params = {
        "q": query,
        "language": "en",
        "pageSize": 5,
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    articles = data.get("articles", [])
    texts = [article["title"] + " " + article["description"] for article in articles if article["description"]]
    return texts

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

def analyze_news(texts):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    preprocessed_news = [preprocess_text(article) for article in texts]
    print("\nPreprocessed News Articles:")
    print(preprocessed_news)
    
    # Generate BERT embeddings
    embeddings = []
    for news in preprocessed_news:
        inputs = tokenizer(news, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
        print(f"\nBERT Embedding for article: {embedding}")

    return embeddings

while True:
    location = input("\nEnter the location to get real-time news (or type 'exit' to quit): ")
    if location.lower() == "exit":
        print("Exiting...")
        break

    news_data = fetch_news(location)
    if not news_data:
        print(f"No news found for '{location}'. Try another location.")
        continue

    analyze_news(news_data)