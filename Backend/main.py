import os
import json
import time
import feedparser
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import gtts
from googletrans import Translator
import urllib.parse
from deep_translator import GoogleTranslator
from dotenv import load_dotenv



# LangChain imports
from langchain_google_genai import GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Load environment variables
load_dotenv()


class NewsAgent:
    def __init__(self):
        print("🚀 Initializing News Agent...")
        self.setup_llm()
        self.setup_embeddings()
        self.setup_vector_store()
        self.test_vector_db()  # Test the vector DB
        self.delete_old_news()  # Delete old news on startup
        self.setup_memory()
        self.setup_search_tools()
        self.setup_tools()
        self.setup_agent()
        self.locations = set()  # Track locations we've already fetched
        print("✅ News Agent initialized and ready!")
    
    def setup_llm(self):
        """Initialize the Gemini model."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
                
            self.llm = GoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=2048
            )
            print("✅ Gemini 1.5 Flash model initialized")
        except Exception as e:
            print(f"❌ Error initializing Gemini model: {e}")
            raise
    
    def setup_embeddings(self):
        """Initialize the embedding model."""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="/app/cache"
            )
            print("✅ Embedding model initialized")
        except Exception as e:
            print(f"❌ Error initializing embedding model: {e}")
            raise
    
    def setup_vector_store(self):
        """Initialize ChromaDB vector store."""
        try:
            self.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embedding_model
            )
            print("✅ Vector store initialized")
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            raise
    
    def test_vector_db(self):
        """Test if the vector database is working properly."""
        try:
            # Check if DB is empty
            db_info = self.vector_store.get()
            print(f"Vector DB contains {len(db_info['ids'])} documents")
            
            if len(db_info['ids']) > 0:
                # Try a simple search
                results = self.vector_store.similarity_search("test", k=1)
                print(f"Test search returned {len(results)} results")
                if results:
                    print(f"Sample document: {results[0].metadata['title']}")
                return True
            else:
                print("Vector DB is empty")
                return False
        except Exception as e:
            print(f"❌ Error testing vector DB: {e}")
            return False
    
    def is_recent_news_available(self, location, max_age_minutes=180):
        """Check if recent news for a location is available in the database."""
        try:
            now = datetime.now()
            # Search for news related to the location
            results = self.vector_store.similarity_search(location, k=10)
            
            # Filter results to those within max_age_minutes
            recent_news = []
            for doc in results:
                metadata = doc.metadata
                if metadata.get('location', '').lower() == location.lower():
                    timestamp_str = metadata.get('timestamp')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if now - timestamp <= timedelta(minutes=max_age_minutes):
                                recent_news.append(doc)
                        except Exception:
                            # Ignore parsing errors
                            continue
            
            print(f"Found {len(recent_news)} recent news items for {location} in database")
            return recent_news
        except Exception as e:
            print(f"❌ Error checking recent news: {e}")
            return []
    
    def delete_old_news(self, max_age_minutes=60):
        """Delete news older than the specified age from the database."""
        try:
            now = datetime.now()
            # Get all documents
            all_docs = self.vector_store.get()
            all_ids = all_docs['ids']
            all_metadatas = all_docs['metadatas']
            
            # Identify documents older than max_age_minutes
            ids_to_delete = []
            for doc_id, metadata in zip(all_ids, all_metadatas):
                timestamp_str = metadata.get('timestamp') if metadata else None
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if now - timestamp > timedelta(minutes=max_age_minutes):
                            ids_to_delete.append(doc_id)
                    except Exception:
                        # Ignore parsing errors
                        continue
            
            # Delete old documents
            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                print(f"✅ Deleted {len(ids_to_delete)} old news items from database")
            
            return len(ids_to_delete)
        except Exception as e:
            print(f"❌ Error deleting old news: {e}")
            return 0
    
    def determine_news_count(self, user_request):
        """Determine how many news articles to fetch based on user request."""
        # Check if user is asking for more news
        more_patterns = ["more news", "additional news", "more articles", "show more", "get more"]
        
        if any(pattern in user_request.lower() for pattern in more_patterns):
            # Check if user specified a number
            number_match = re.search(r'(\d+)\s+(more|additional|extra)', user_request.lower())
            if number_match:
                try:
                    count = int(number_match.group(1))
                    # Cap at a reasonable maximum
                    return min(count, 20)
                except ValueError:
                    pass
            
            return 15  # Return more news if requested without specific number
        else:
            return 5   # Default number of news
    
    def setup_memory(self):
        """Initialize conversation memory."""
        try:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=4000,  # Increased token limit for better context retention
                return_messages=True,
                memory_key="chat_history",
                input_key="input",      # Explicitly define input key
                output_key="output"     # Explicitly define output key
            )
            print("✅ Conversation memory initialized")
        except Exception as e:
            print(f"❌ Error initializing memory: {e}")
            raise
    
    def setup_search_tools(self):
        """Set up search tools."""
        try:
            # Setup DuckDuckGo search
            self.ddg_wrapper = DuckDuckGoSearchAPIWrapper(
                time="d",  # Search for content from the past day
                max_results=5
            )
            
            # Setup DuckDuckGo news search
            self.ddg_news_wrapper = DuckDuckGoSearchAPIWrapper(
                time="d",  # Search for content from the past day
                max_results=5
            )
            
            print("✅ Search tools initialized")
        except Exception as e:
            print(f"❌ Error initializing search tools: {e}")
            raise
    
    def setup_tools(self):
        """Set up tools for the agent."""
        self.tools = [
            Tool(
                name="FetchNews",
                func=self.fetch_city_news,
                description="Fetches the latest news for a specific city or location. Input should be the name of the city or 'city, number' to specify how many articles to fetch."
            ),
            Tool(
                name="SearchNewsArticle",
                func=self.search_news_article,
                description="Searches for news articles on a specific topic or title and returns summaries. Input should be the topic or title to search for."
            ),
            Tool(
                name="GetMoreInfoOnNews",
                func=self.get_more_info_on_news,
                description="Gets more detailed information about a specific news story. Input should be the news title or topic you want more information about."
            ),
            Tool(
                name="GetArticleContent",
                func=self.get_article_content,
                description="Gets the content of a news article from a URL. Input should be the URL of the article."
            ),
            Tool(
                name="SummarizeText",
                func=self.summarize_text,
                description="Summarizes a text. Input should be the text to summarize."
            ),
            Tool(
                name="TextToSpeech",
                func=self.text_to_speech,
                description="Converts text to speech in a specified language. Input should be a JSON string with 'text' and 'lang' keys."
            ),
            Tool(
                name="TranslateText",
                func=self.translate_text,
                description="Translates text to a specified language. Input should be a JSON string with 'text' and 'lang' keys."
            ),
            Tool(
                name="SearchNewsInDB",
                func=self.search_news_in_db,
                description="Searches for news in the database. Input should be the search query."
            ),
            Tool(
                name="GetRecentNewsFromDB",
                func=self.get_recent_news_from_db,
                description="Gets recent news for a location from the database. Input should be the location name."
            )
        ]
        print("✅ Agent tools initialized")
    
    def setup_agent(self):
        """Set up the LangChain agent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that specializes in providing location-specific news Developed by GFG-KIIT AI/ML Team.
            You can fetch news, search for articles, get more information on specific news stories, summarize text, translate content, and convert text to speech.
            Always try to understand what location the user is asking about and provide relevant news.
            If you're not sure about a location, ask for clarification.
            
            IMPORTANT: Maintain conversation context. When the user asks follow-up questions about previously mentioned news articles, 
            use your memory of the conversation to understand which article they're referring to. If they ask for more details about a 
            news story you've mentioned, use the GetMoreInfoOnNews tool with the appropriate title.
            
            When providing news:
            1. Always ensure you're providing the most recent news (from today if possible)
            2. First check if recent news is available in the database before fetching from the web
            3. If a user asks for more information about a specific news story, use the GetMoreInfoOnNews tool
            4. Always include relevant links when providing detailed information about news
            5. Summarize news articles in a concise and informative way
            6. If a user asks for more news, provide additional articles (up to 15)
            7. Remember which news articles you've already mentioned in the conversation
            
            You have access to the following tools:
            
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Chat History: {chat_history}
            """),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Return intermediate steps for better debugging
        )
        print("✅ Agent executor initialized")

    
    def get_recent_news_from_db(self, location):
        """Gets recent news for a location from the database."""
        try:
            recent_news = self.is_recent_news_available(location)
            
            if not recent_news:
                return f"No recent news found in database for {location}. Try fetching fresh news."
            
            response = f"📰 Recent News from {location} (from database):\n\n"
            for i, doc in enumerate(recent_news, 1):
                metadata = doc.metadata
                response += f"{i}. {metadata.get('title', 'Unknown Title')}\n"
                response += f"   Source: {metadata.get('source', 'Unknown Source')}\n"
                response += f"   Published: {metadata.get('date', 'Unknown Date')}\n"
                response += f"   Link: {metadata.get('link', 'No Link Available')}\n"
                
                # Extract summary from content
                content = doc.page_content
                summary_match = re.search(r"SUMMARY: (.*?)(?:CONTENT:|$)", content, re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1).strip()
                    response += f"   Summary: {summary}\n"
                
                response += "\n"
            
            return response
        except Exception as e:
            print(f"❌ Error getting recent news from DB: {e}")
            return f"Error retrieving recent news for {location} from database."
    
    def search_news_article(self, query):
        """Search for news articles on a specific topic using DuckDuckGo News."""
        try:
            print(f"🔍 Searching for news articles on: {query}")
            
            # Parse input for number of results if provided
            parts = query.split(',')
            search_query = parts[0].strip()
            max_results = 5
            
            if len(parts) > 1:
                try:
                    max_results = int(parts[1].strip())
                    max_results = min(max_results, 20)  # Cap at 20 results
                except ValueError:
                    pass
            
            # Use DuckDuckGo search with news-specific query
            search_results = self.ddg_news_wrapper.results(f"{search_query} news", max_results=max_results)
            
            if not search_results:
                return f"No news articles found for: {search_query}"
            
            # Process search results
            articles = []
            for i, result in enumerate(search_results[:max_results]):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No snippet")
                published_date = result.get("published", datetime.now().strftime("%a, %d %b %Y %H:%M:%S"))
                source = result.get("source", "Unknown source")
                
                # Create article object
                article = {
                    "title": title,
                    "source": source,
                    "link": link,
                    "published": published_date,
                    "snippet": snippet,
                    "query": search_query
                }
                
                articles.append(article)
                
                # Store in vector database for RAG
                self.store_article_in_db(article)
            
            # Format response
            response = f"📰 Latest News Articles on '{search_query}':\n\n"
            for i, article in enumerate(articles, 1):
                response += f"{i}. {article['title']}\n"
                response += f"   Source: {article['source']}\n"
                response += f"   Published: {article['published']}\n"
                response += f"   Link: {article['link']}\n"
                response += f"   Summary: {article['snippet']}\n\n"
            
            return response
            
        except Exception as e:
            print(f"❌ Error searching for news articles: {e}")
            return f"Error searching for news articles on '{query}': {str(e)}"
    
    def get_more_info_on_news(self, news_title):
        """Gets more detailed information about a specific news story."""
        try:
            print(f"🔍 Getting more information on: {news_title}")
            
            # First, search for the news in our database
            db_results = self.search_news_in_db(news_title, k=1)
            
            # If we found something relevant in the database
            if "No relevant news found" not in db_results:
                # Extract the URL from the database results
                url_match = re.search(r"Link: (https?://[^\s]+)", db_results)
                if url_match:
                    article_url = url_match.group(1)
                    
                    # Get the full content of the article
                    content = self.get_article_content(article_url)
                    
                    # Summarize the content
                    summary = self.summarize_text(content)
                    
                    return f"📰 More Information on '{news_title}':\n\n{summary}\n\nSource: {article_url}"
            
            # If we didn't find anything in the database or couldn't extract the URL,
            # search for the news using DuckDuckGo
            search_results = self.ddg_wrapper.results(f"{news_title} latest news", max_results=5)
            
            if not search_results:
                return f"Could not find more information on: {news_title}"
            
            # Get the first result
            result = search_results[0]
            article_url = result.get("link")
            
            if not article_url:
                return f"Could not find a relevant article for: {news_title}"
            
            # Get the content of the article
            content = self.get_article_content(article_url)
            
            # Summarize the content
            summary = self.summarize_text(content)
            
            # Store in vector database for future reference
            self.store_article_in_db({
                "title": news_title,
                "link": article_url,
                "content": content,
                "summary": summary,
                "source": result.get("source", "Unknown source"),
                "published": datetime.now().strftime("%a, %d %b %Y")
            })
            
            return f"📰 More Information on '{news_title}':\n\n{summary}\n\nSource: {article_url}"
            
        except Exception as e:
            print(f"❌ Error getting more information: {e}")
            return f"Error getting more information on '{news_title}': {str(e)}"
    
    def get_article_content(self, url):
        """Extract content from a news article URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Check if URL is valid
            if not url.startswith('http'):
                return "Invalid URL. Please provide a URL starting with http:// or https://"
            
            # Send request
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and nav elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try to find the main content
            main_content = None
            
            # Look for article tag
            article = soup.find('article')
            if article:
                main_content = article
            
            # Look for main tag if article not found
            if not main_content:
                main_tag = soup.find('main')
                if main_tag:
                    main_content = main_tag
            
            # Look for div with content-related class names
            if not main_content:
                content_div = soup.find('div', class_=lambda c: c and any(x in c.lower() for x in ['content', 'article', 'story', 'entry', 'post']))
                if content_div:
                    main_content = content_div
            
            # Extract text from main content or fallback to body
            if main_content:
                paragraphs = main_content.find_all('p')
            else:
                paragraphs = soup.find_all('p')
            
            # Join paragraphs
            content = '\n\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40])
            
            # If content is too short, try a different approach
            if len(content) < 200:
                # Get all text from body
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n')
                    
                    # Clean up content
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    content = '\n'.join(lines)
            
            # If still no content, return error
            if not content or len(content) < 100:
                return "Could not extract meaningful content from the article."
            
            # Truncate if too long
            if len(content) > 8000:
                content = content[:8000] + "...[content truncated]"
            
            return content
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching article: {str(e)}"
        except Exception as e:
            return f"Error extracting content: {str(e)}"
    
    def summarize_text(self, text):
        """Summarize text using the LLM."""
        try:
            if not text or len(text) < 100:
                return "Text is too short to summarize."
            
            # Truncate text if it's too long
            if len(text) > 10000:
                text = text[:10000] + "...[content truncated]"
            
            prompt = f"""
            Summarize the following news article in a concise way (3-5 sentences), highlighting the key points:
            
            {text}
            
            Summary:
            """
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"❌ Error summarizing text: {e}")
            return "Could not generate summary due to an error."
    
    def fetch_city_news(self, city_input, max_articles=5):
        """Fetch news for a specific city using Google News RSS first, then enhance with search."""
        # Parse input for city and optional count
        parts = city_input.split(',')
        city = parts[0].strip()
        
        if len(parts) > 1:
            try:
                max_articles = int(parts[1].strip())
                max_articles = min(max_articles, 20)  # Cap at 20 articles
            except ValueError:
                pass
        
        print(f"🔍 Fetching {max_articles} news articles for: {city}")
        
        # Check if we have recent news in the database
        recent_news = self.is_recent_news_available(city)
        if recent_news and len(recent_news) >= max_articles:
            print(f"✅ Found {len(recent_news)} recent news items in database for {city}")
            response = f"📰 Latest News from {city} (from database):\n\n"
            for i, doc in enumerate(recent_news[:max_articles], 1):
                metadata = doc.metadata
                response += f"{i}. {metadata.get('title', 'Unknown Title')}\n"
                response += f"   Source: {metadata.get('source', 'Unknown Source')}\n"
                response += f"   Published: {metadata.get('date', 'Unknown Date')}\n"
                response += f"   Link: {metadata.get('link', 'No Link Available')}\n"
                
                # Extract summary from content
                content = doc.page_content
                summary_match = re.search(r"SUMMARY: (.*?)(?:CONTENT:|$)", content, re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1).strip()
                    response += f"   Summary: {summary}\n"
                
                response += "\n"
            
            return response
        
        # Clean the city name to avoid URL issues
        clean_city = city.strip().replace("\n", "").replace("\r", "")
        encoded_city = urllib.parse.quote(clean_city)
        
        try:
            # First get news from Google News RSS
            rss_url = f"https://news.google.com/rss/search?q={encoded_city}+when:1d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                return f"No news found for {city}"
            
            # Process articles from RSS feed
            articles = []
            for entry in feed.entries[:max_articles]:
                # Extract title and source
                title_parts = entry.title.split(" - ")
                title = title_parts[0].strip() if len(title_parts) > 1 else entry.title.strip()
                source = title_parts[-1].strip() if len(title_parts) > 1 else "Unknown"
                
                # Get the article link
                google_news_link = entry.link
                
                # Extract publication date
                published_date = entry.get("published", datetime.now().strftime("%a, %d %b %Y"))
                
                print(f"📰 Found news: {title}")
                print(f"🔍 Searching for more details about: {title}")
                
                # Now search for more details about this specific news
                try:
                    search_results = self.ddg_wrapper.results(f"{title} {city} news", max_results=3)
                    
                    if search_results:
                        # Get the first result
                        result = search_results[0]
                        article_url = result.get("link")
                        
                        # Get the content of the article
                        content = self.get_article_content(article_url)
                        
                        # Summarize the content
                        summary = self.summarize_text(content)
                    else:
                        article_url = google_news_link
                        content = ""
                        summary = "No additional details available."
                except Exception as e:
                    print(f"❌ Error getting more details: {e}")
                    article_url = google_news_link
                    content = ""
                    summary = "Could not retrieve additional details due to an error."
                
                # Create article object
                article = {
                    "title": title,
                    "source": source,
                    "link": article_url,
                    "published": published_date,
                    "location": city,
                    "summary": summary,
                    "content": content if 'content' in locals() else ""
                }
                
                articles.append(article)
                
                # Store in vector database for RAG
                self.store_article_in_db(article)
            
            # Add location to tracked locations
            self.locations.add(city.lower())
            
            # Format response
            response = f"📰 Latest News from {city}:\n\n"
            for i, article in enumerate(articles, 1):
                response += f"{i}. {article['title']}\n"
                response += f"   Source: {article['source']}\n"
                response += f"   Published: {article['published']}\n"
                response += f"   Link: {article['link']}\n"
                response += f"   Summary: {article['summary']}\n\n"
            
            return response
        
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return f"Error fetching news for {city}: {str(e)}"
    
    def store_article_in_db(self, article):
        """Store an article in the vector database."""
        try:
            # Create document text
            doc_text = f"""
            TITLE: {article.get('title', 'Unknown Title')}
            SOURCE: {article.get('source', 'Unknown Source')}
            PUBLISHED: {article.get('published', datetime.now().strftime('%a, %d %b %Y'))}
            LOCATION: {article.get('location', 'Unknown Location')}
            LINK: {article.get('link', 'No Link Available')}
            SUMMARY: {article.get('summary', article.get('snippet', 'No Summary Available'))}
            CONTENT: {article.get('content', 'No Content Available')}
            """
            
            # Add metadata
            metadata = {
                "title": article.get('title', 'Unknown Title'),
                "source": article.get('source', 'Unknown Source'),
                "location": article.get('location', 'Unknown Location'),
                "date": article.get('published', datetime.now().strftime('%a, %d %b %Y')),
                "link": article.get('link', 'No Link Available'),
                "type": "news",
                "timestamp": datetime.now().isoformat()  # Add timestamp for recency filtering
            }
            
            # Create document
            document = Document(page_content=doc_text, metadata=metadata)
            
            # Add to vector store - this automatically persists the data
            self.vector_store.add_documents([document])
            
            # Verify storage
            print(f"✅ Stored article in vector database: {article.get('title', 'Unknown Title')}")
            try:
                db_info = self.vector_store.get()
                print(f"   Current DB size: {len(db_info['ids'])} documents")
            except:
                print("   Could not verify DB size")
            
            return True
        except Exception as e:
            print(f"❌ Error storing article: {e}")
            print(f"Article data: {article}")
            return False
    
    def text_to_speech(self, input_json):
        """Convert text to speech in the specified language."""
        try:
            # Parse input JSON
            try:
                data = json.loads(input_json)
                text = data.get("text", "")
                lang = data.get("lang", "en")
            except json.JSONDecodeError:
                # If not valid JSON, assume it's just text
                text = input_json
                lang = "en"
            
            if not text:
                return "No text provided for speech conversion."
            
            # Get supported languages
            supported_languages = gtts.lang.tts_langs()
            
            if lang not in supported_languages:
                return f"Language '{lang}' is not supported for text-to-speech."
            
            # Generate speech
            output_file = f"speech_{int(time.time())}.mp3"
            tts = gtts.gTTS(text=text, lang=lang, slow=False)
            tts.save(output_file)
            
            return f"Successfully converted text to speech in {supported_languages[lang]}."
        except Exception as e:
            print(f"❌ Error in text-to-speech: {e}")
            return f"Error in text-to-speech: {str(e)}"
    

    def translate_text(self, input_json):
        """Translate text to the specified language."""
        try:
            # Parse input JSON
            try:
                data = json.loads(input_json)
                text = data.get("text", "")
                lang = data.get("lang", "en")
            except json.JSONDecodeError:
                # If not valid JSON, assume format is "text|lang"
                parts = input_json.split("|")
                text = parts[0]
                lang = parts[1] if len(parts) > 1 else "en"
            
            if not text:
                return "No text provided for translation."
            
            # Translate text using deep-translator
            translator = GoogleTranslator(source='auto', target=lang)
            translated_text = translator.translate(text)
            
            return f"Translated text: {translated_text}"
        except Exception as e:
            print(f"❌ Error in translation: {e}")
            return f"Error in translation: {str(e)}"



    
    def search_news_in_db(self, query, k=3):
        """Search for news in the vector database with recency filtering."""
        try:
            # Get current date
            current_date = datetime.now()
            
            # First, perform the similarity search
            results = self.vector_store.similarity_search(query, k=k*2)  # Get more results than needed for filtering
            
            if not results:
                return "No relevant news found in the database."
            
            # Filter for recent news (prioritize news from the last 24 hours)
            recent_results = []
            older_results = []
            
            for doc in results:
                metadata = doc.metadata
                timestamp_str = metadata.get("timestamp")
                
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        # If news is from the last 24 hours
                        if current_date - timestamp <= timedelta(days=1):
                            recent_results.append(doc)
                        else:
                            older_results.append(doc)
                    except (ValueError, TypeError):
                        older_results.append(doc)
                else:
                    older_results.append(doc)
            
            # Combine recent and older results, prioritizing recent ones
            filtered_results = recent_results + older_results
            
            # Limit to the requested number of results
            filtered_results = filtered_results[:k]
            
            if not filtered_results:
                return "No relevant news found in the database."
            
            response = "📰 Related News from Database:\n\n"
            for i, doc in enumerate(filtered_results, 1):
                metadata = doc.metadata
                response += f"{i}. {metadata.get('title', 'Unknown Title')}\n"
                response += f"   Source: {metadata.get('source', 'Unknown Source')}\n"
                response += f"   Location: {metadata.get('location', 'Unknown Location')}\n"
                response += f"   Published: {metadata.get('date', 'Unknown Date')}\n"
                response += f"   Link: {metadata.get('link', 'No Link Available')}\n\n"
            
            return response
        except Exception as e:
            print(f"❌ Error searching news in DB: {e}")
            return "Error searching the news database."
    
    def extract_locations(self, query):
        """Extract potential location names from the query."""
        try:
            prompt = f"""
            Extract any city or country names from this text. Return ONLY the names separated by commas, or 'None' if no locations are found:
            
            Text: {query}
            """
            
            response = self.llm.invoke(prompt)
            locations = [loc.strip() for loc in response.strip().split(',') if loc.strip().lower() != 'none']
            return locations
        except Exception:
            # Fallback to simple keyword extraction
            common_cities = ["new york", "london", "tokyo", "paris", "delhi", "mumbai", "kolkata", "bangalore", "bhubaneswar"]
            found = []
            for city in common_cities:
                if city.lower() in query.lower():
                    found.append(city)
            return found
    
    def process_query(self, query):
        """Process a user query through the agent."""
        # Clean up old news first
        self.delete_old_news()
        
        # Get conversation history to provide context
        chat_history = self.get_conversation_context()
        
        # Determine how many news to fetch
        news_count = self.determine_news_count(query)
        
        # Check if query contains a location
        potential_locations = self.extract_locations(query)
        
        # Check if user is asking for more details about a specific news
        is_asking_for_details = any(pattern in query.lower() for pattern in 
                                ["more details", "tell me more about", "more information on", 
                                    "details on", "what about", "tell me about"])
        
        # If asking for details about specific news, try to extract the news title from context
        if is_asking_for_details and not any(word in query.lower() for word in ["news", "article"]):
            # Try to extract news title from the query or recent conversation
            news_title = self.extract_news_title_from_context(query, chat_history)
            if news_title:
                print(f"📝 Extracted news title from context: {news_title}")
                # Append the extracted title to the query for clarity
                query = f"{query} about '{news_title}'"
        
        # For location-based queries
        for location in potential_locations:
            # Check if we have recent news in the database
            recent_news = self.is_recent_news_available(location)
            
            # If user wants more news or we don't have recent news, fetch from web
            if not recent_news or "more" in query.lower():
                if location.lower() not in [loc.lower() for loc in self.locations]:
                    print(f"🔄 Detected new location: {location}. Fetching news...")
                    self.fetch_city_news(f"{location}, {news_count}")
        
        # Process through the agent with enhanced context
        try:
            chat_history = self.get_conversation_context()
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history  # This will be included in the system message
            })
            return response["output"]
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

    def get_conversation_context(self):
        """Get formatted conversation history for context."""
        try:
            # Get messages from memory
            messages = self.memory.chat_memory.messages
            
            if not messages:
                return []
            
            return messages
        except Exception as e:
            print(f"❌ Error retrieving conversation context: {e}")
            return []
        
    def extract_news_title_from_context(self, query, chat_history):
        """Extract relevant news title from conversation context or query."""
        try:
            # First, check if there are any news titles in the recent AI messages
            recent_ai_messages = [msg.content for msg in chat_history[-4:] if hasattr(msg, 'type') and msg.type == 'ai']
            
            # Combine recent AI messages
            context_text = " ".join(recent_ai_messages)
            
            # Look for news titles in the format typically used in our responses
            title_matches = re.findall(r'\d+\.\s+(.*?)\n', context_text)
            
            if title_matches:
                # Use the LLM to determine which title is most relevant to the query
                titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(title_matches)])
                
                prompt = f"""
                Given the user query and the list of recently mentioned news titles, which title is the user most likely referring to?
                Return ONLY the title, or "None" if none seem relevant.
                
                User query: {query}
                
                Recently mentioned titles:
                {titles_text}
                """
                
                response = self.llm.invoke(prompt).strip()
                
                if response and response.lower() != "none":
                    return response
            
            # If we couldn't find a title from context, try to extract it from the query
            # This is a fallback for explicit mentions
            query_words = query.lower().split()
            for i, word in enumerate(query_words):
                if word in ["about", "regarding", "concerning", "on"]:
                    if i+1 < len(query_words):
                        potential_title = " ".join(query_words[i+1:])
                        # Remove quotes if present
                        potential_title = potential_title.strip('"\'')
                        if len(potential_title) > 3:  # Minimum length check
                            return potential_title
            
            return None
        except Exception as e:
            print(f"❌ Error extracting news title from context: {e}")
            return None



def main():
    print("=" * 50)
    print("🌍 Location-Specific News Agent")
    print("=" * 50)
    print("Initializing system...")
    
    agent = NewsAgent()
    
    print("\nChat with the news agent! Type 'exit' to quit.")
    print("Example: 'What's happening in Delhi today?'")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the news agent. Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = agent.process_query(user_input)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()
