import json
import os
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is missing. Set it in the .env file.")

# Initialize the language model
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up ChromaDB for vector storage
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Set up web search tool using DuckDuckGo
search_tool = DuckDuckGoSearchRun(max_results=3)

# Memory to retain conversation summaries
memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True, memory_key="history")

# Prompt template for news summarization
news_prompt = PromptTemplate(
    input_variables=["history", "input", "local_data", "web_data"],
    template="""
    You are a helpful news summarizer assistant. The user asked: "{input}"
    Here’s the relevant data from our local database:
    \"{local_data}\"
    Here’s additional information from a web search:
    \"{web_data}\"
    Combine and summarize this information into a concise, natural response (3-5 sentences) that addresses the user's query. Include key details from both sources and ensure the response is coherent and informative. Cite sources where appropriate.
    Conversation history:
    {history}
    """
)

# Prompt template for general user queries
general_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are a helpful assistant.
    The user asked: "{input}"
    Provide a detailed and accurate answer based on your knowledge.
    Conversation history:
    {history}
    """
)

# Chains for response generation
news_chain = news_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()

# Load mock database from JSON file
def load_mock_database():
    """Load entries from the mock JSON database."""
    try:
        with open("mock_database.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

# Add documents to the vector store
def populate_mock_database():
    """Populate ChromaDB with entries from the mock database."""
    mock_database = load_mock_database()
    if not mock_database:
        print("No mock database entries found.")
        return
    for entry in mock_database:
        metadata_str = "\n".join([f"{k}: {v}" for k, v in entry.get("metadata", {}).items()])
        content = f"Category: {entry['category']}\n{entry['content']}"
        if metadata_str:
            content += f"\n{metadata_str}"
        vector_store.add_documents([Document(page_content=content)])
    print("Mock database populated.")

# Retrieve past conversation history
def retrieve_past_conversations():
    """Fetch summarized past conversation history."""
    return memory.load_memory_variables({}).get("history", "")

# Search ChromaDB for relevant local information
def get_response_from_local_db(query, score_threshold=0.75):
    """Search vector DB for local content matching the user query."""
    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=3)
        filtered = [
            doc.page_content for doc, score in results_with_scores
            if "Category:" in doc.page_content and score <= score_threshold
        ]
        return "\n\n".join(filtered) if filtered else None
    except Exception as e:
        print(f"Error searching local DB: {e}")
        return None

# Combine and summarize local and web data
def summarize_combined_data(user_query, local_data, web_data):
    """Generate summarized output from local and web data sources."""
    conversation_history = retrieve_past_conversations()
    response = news_chain.invoke({
        "input": user_query,
        "history": conversation_history,
        "local_data": local_data or "No relevant local data found.",
        "web_data": web_data or "No additional web data found."
    })
    return response

# Answer general-purpose queries using the LLM
def answer_general_query(user_query):
    """Answer general queries when local data is insufficient."""
    conversation_history = retrieve_past_conversations()
    response = general_chain.invoke({
        "input": user_query,
        "history": conversation_history
    })
    return response

# Main chatbot interaction loop
def run_chatbot():
    """Run the interactive chatbot loop."""
    print("Chatbot: Hello! Ask me about news topics. Type 'more about [topic]' for detailed info or 'exit' to quit.")
    populate_mock_database()

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            print("Chatbot: Please say something.")
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        if user_input.lower().startswith("more about"):
            topic = user_input[10:].strip()
            if not topic:
                print("Chatbot: Please specify a topic after 'more about'.")
                continue
            local_response = get_response_from_local_db(topic)
            try:
                web_response = search_tool.run(topic + " news")
            except Exception as e:
                print(f"Error with DuckDuckGo search: {e}")
                web_response = None
            response = summarize_combined_data(topic, local_response, web_response)
        else:
            local_response = get_response_from_local_db(user_input)
            if local_response and len(local_response.strip()) > 10:
                response = summarize_combined_data(user_input, local_response, None)
            else:
                response = answer_general_query(user_input)

        memory.save_context({"input": user_input}, {"output": response})
        print("Chatbot:", response)

if __name__ == "__main__":
    run_chatbot()
