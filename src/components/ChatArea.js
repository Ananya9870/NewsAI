import React, { useState, useEffect, useRef } from "react";
import { useUserPreferences } from "../context/UserPreferences";
import { searchNews } from "../utils/database";
import { languages } from "../utils/languages";

function ChatArea() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const { language, location } = useUserPreferences();
  const chatHistoryRef = useRef(null);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  // Add a welcome message when component mounts
  useEffect(() => {
    const currentLanguage = languages[language] || "English";
    const welcomeMessage = {
      text: `Welcome! I can provide news summaries in ${currentLanguage}${location ? ` for ${location.name}` : ''}. How can I help you today?`,
      sender: "bot",
    };
    setMessages([welcomeMessage]);
  }, [language, location]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    
    // Search the news database for relevant articles
    const searchResults = searchNews(input, language);
    
    // Generate a response based on search results
    let botResponse;
    if (searchResults.length > 0) {
      // Format a response with the search results
      const summary = searchResults.map(article => 
        `- ${article.title}: ${article.summary}`
      ).join('\n');
      
      botResponse = {
        text: `Here's what I found about "${input}":\n\n${summary}`,
        sender: "bot",
      };
    } else {
      // No results found
      botResponse = {
        text: `I couldn't find any specific news about "${input}". Would you like to try a different query?`,
        sender: "bot",
      };
    }
    
    setMessages((prev) => [...prev, botResponse]);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  const handleMicClick = () => {
    console.log("Microphone clicked! Implement speech-to-text here.");
    // To implement speech-to-text, you can use the Web Speech API:
    // const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    // recognition.onresult = (event) => {
    //   const transcript = event.results[0][0].transcript;
    //   setInput(transcript);
    // };
    // recognition.start();
  };

  return (
    <div className="flex flex-col h-full w-full max-w-3xl mx-auto">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-white">News Summarizer</h1>
        <p className="text-[#8e8ea0]">
          {location ? `Local news for ${location.name}` : 'Set your location for local news'} 
          {language && ` • ${languages[language]}`}
        </p>
      </div>
      
      <div
        ref={chatHistoryRef}
        className="flex-1 overflow-y-auto bg-[#1e2129] p-4 space-y-4 custom-scrollbar"
      >
        {messages.length === 0 ? (
          <div className="text-center text-[#e5e7eb] opacity-70 mt-20">
            Start asking about the news!
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg ${
                msg.sender === "user"
                  ? "bg-[#343541] text-white ml-auto max-w-[80%]"
                  : "bg-[#444654] text-white max-w-[80%]"
              }`}
            >
              {msg.text.split('\n').map((line, i) => (
                <React.Fragment key={i}>
                  {line}
                  {i < msg.text.split('\n').length - 1 && <br />}
                </React.Fragment>
              ))}
            </div>
          ))
        )}
      </div>
      <div className="mt-4 flex items-center p-2 bg-[#40414f] rounded-lg shadow-md w-full">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          className="flex-1 p-2 bg-transparent text-white placeholder-[#8e8ea0] focus:outline-none"
          placeholder="Type your prompt..."
        />
        <button
          onClick={handleMicClick}
          className="p-2 text-white rounded-md hover:bg-[#4a4e5a] transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
        </button>
        <button
          onClick={handleSend}
          className="p-2 bg-[#3b82f6] text-white rounded-md hover:bg-[#2563eb] transition-colors"
        >
          Send
        </button>
      </div>

      {/* Inline Custom Scrollbar Styles */}
      <style>{`
        .custom-scrollbar {
          scrollbar-width: thin; /* Firefox */
          scrollbar-color: #4a4e5a #2a2d37; /* Firefox */
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #2a2d37;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #4a4e5a;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #5a5e6a;
        }
        .custom-scrollbar {
          overflow-y: auto !important;
        }
      `}</style>
    </div>
  );
}

export default ChatArea;
