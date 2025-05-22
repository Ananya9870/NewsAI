## NewsAI 
🌍 Location-Specific News Agent
A powerful AI-driven agent that provides real-time, location-specific news, enriched with capabilities such as summarization, translation, speech synthesis, and intelligent memory-based interactions. It leverages Google Gemini, LangChain, ChromaDB, and DuckDuckGo to deliver contextual news experiences.


🚀 Features
🔎 City-Based News Search: Just ask “What’s the latest in Delhi?” and get today’s news.

🧠 Memory-Powered Conversations: Follows up intelligently when you ask for “more info on the first one.”

💬 News Summarization: Extracts and summarizes detailed articles.

🌐 Multilingual Translation: Translates articles to your preferred language.

🔊 Text-to-Speech: Listen to news headlines and summaries in your language.

🗃️ Vector Database: Stores and retrieves recent news for enhanced performance.

🔁 Automatic Old News Cleanup: Keeps your database fresh.

🧰 Built With
LangChain
Google Gemini (Generative AI)
Chroma Vector DB
HuggingFace Embeddings
DuckDuckGo Search API Wrapper
Google News RSS
BeautifulSoup
Deep Translator
gTTS
playsound3

Set up a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


Set your API Key
Replace the placeholder inside main.py with your actual Google API key:
GOOGLE_API_KEY = "your-api-key-here"
