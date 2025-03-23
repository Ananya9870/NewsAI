import React, { useState, useEffect, useRef } from "react";

function ChatArea() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const chatHistoryRef = useRef(null);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    const botResponse = {
      text: `Summary for "${input}": Here's what I found...`,
      sender: "bot",
    };
    setMessages((prev) => [...prev, botResponse]);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className="flex flex-col h-full w-full max-w-3xl mx-auto">
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
              {msg.text}
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
          onClick={handleSend}
          className="p-2 bg-[#10a37f] text-white rounded-md hover:bg-[#0d8c6b] transition-colors"
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
        /* Ensure the scrollbar is visible even if the browser doesn't support customization */
        .custom-scrollbar {
          overflow-y: auto !important;
        }
      `}</style>
    </div>
  );
}

export default ChatArea;
