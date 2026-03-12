NewsAI: AI-Powered News Analytics Dashboard
NewsAI is a sophisticated news aggregation and analysis platform that transforms raw news data into actionable insights. By leveraging Large Language Models (LLMs) and advanced Natural Language Processing (NLP), the platform provides real time summaries, sentiment tracking, and interactive visualizations to help users stay informed efficiently.

🚀 Key Features
Real-Time News Aggregation: Automatically fetches the latest headlines across multiple categories and sources.

Gemini-Powered Insights: Utilizes Google Gemini Pro to generate concise, high quality summaries of complex news articles.

BERT-Based Sentiment Analysis: Employs a fine-tuned BERT model to classify the emotional tone of news (Positive, Negative, or Neutral).

Interactive Dashboard: A clean, responsive UI built with Streamlit for seamless data exploration and filtering.

Trend Visualization: Visual representations of news volume and sentiment shifts over time.

🛠️ Tech Stack
Frontend: Streamlit (for a fast, interactive web interface)

Backend Framework: FastAPI (for high-performance API management)

LLM/Generative AI: Google Gemini Pro (for summarization and insight generation)

NLP/Machine Learning: BERT (Bidirectional Encoder Representations from Transformers)

Database: MongoDB (for scalable storage of news metadata and analysis)

Programming Language: Python 3.x

🔌 APIs & Models
Google Gemini API: Powers the intelligent summary engine to distill long-form articles into key bullet points.

HuggingFace Transformers: Used to implement the BERT model for state-of-the-art sentiment classification.

News API / GNews: (Or your specific provider) Used to fetch the latest global news data.

⚙️ How It Works
Data Ingestion: The system polls news APIs to retrieve the latest articles based on user interests or trending topics.

Processing Pipeline:

Summarization: The content is sent to Gemini Pro to create a "Too Long; Didn't Read" (TL;DR) version.

Sentiment Scoring: The BERT model analyzes the text to determine the prevailing sentiment.

Storage: Processed insights are stored in MongoDB to allow for historical analysis and faster retrieval.

Presentation: The Streamlit dashboard fetches data from the backend/database and renders it into interactive charts and news cards.

📈 Benefits
Efficiency: Reduces the time spent reading multiple news sources by providing centralized, summarized content.

Objectivity: Sentiment analysis helps users identify potential media bias or market mood at a glance.

Scalability: The architecture supports adding new models or data sources without disrupting the core UI.

📋 Installation & Setup
Clone the Repository:

Bash
git clone https://github.com/Ananya9870/NewsAI.git
cd NewsAI
Create a Virtual Environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

Bash
pip install -r requirements.txt
Set Up Environment Variables:
Create a .env file in the root directory and add your keys:

Code snippet
GEMINI_API_KEY=your_google_gemini_key
NEWS_API_KEY=your_news_api_key
MONGO_URI=your_mongodb_connection_string
Run the Application:

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request for any features or bug fixes.
