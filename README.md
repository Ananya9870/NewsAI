# 📰 NewsAI: AI-Powered News Analytics Dashboard

## 🚀 Project Overview
**NewsAI** is built for anyone who feels overwhelmed by the endless cycle of news. Instead of scrolling through dozens of long articles, this platform does the heavy lifting for you. By combining the power of Large Language Models (LLMs) and Natural Language Processing (NLP), it scans the latest headlines, breaks them down into quick, easy-to-read summaries, and even tracks the "vibe" or emotional tone of the news. It’s essentially a smart news assistant that helps you stay informed and spot trends without the information overload.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Real-Time Aggregation** | Automatically fetches latest headlines across global categories. |
| **Gemini Insights** | Uses **Google Gemini Pro** to generate "TL;DR" summaries of long articles. |
| **Sentiment Analysis** | Employs a fine-tuned **BERT** model to classify news as Positive, Negative, or Neutral. |
| **Interactive UI** | A responsive, data-driven dashboard built. |
| **Trend Tracking** | Visualizes news volume and sentiment shifts through interactive charts. |

---

## 🛠️ Tech Stack

### **Core Technologies**
* **Backend:** `FastAPI` (High-performance API layer)
* **Database:** `MongoDB` (Scalable storage for news metadata)

### **AI & Machine Learning**
* **LLM:** `Google Gemini Pro` (Summarization & Insight generation)
* **NLP:** `BERT` (State-of-the-art sentiment classification via HuggingFace)
* **Environment:** `Python 3.10+`

---

## ⚙️ How It Works

1. **Ingestion:** The system polls News APIs or RSS feeds to retrieve the latest global headlines.
2. **Processing Pipeline:**
    * **Summarization:** Articles are processed by **Gemini Pro** to distill key points into a concise summary.
    * **Sentiment Scoring:** The **BERT** model analyzes the text to determine the prevailing emotional tone (Positive/Negative/Neutral).
3. **Storage:** Metadata and analysis results are indexed in **MongoDB** for historical tracking and fast retrieval.
   
---

## 🔌 API & Model Integration

* **Google Gemini API:** Leveraged for advanced reasoning and natural language generation.
* **HuggingFace Transformers:** Hosts the BERT model for deep-learning-based text analysis.
* **News API / GNews:** Provides the raw data stream for global news coverage.

---

## 🚀 Getting Started

### **1. Clone & Navigate**
```bash
git clone [https://github.com/Ananya9870/NewsAI.git](https://github.com/Ananya9870/NewsAI.git)
cd NewsAI
