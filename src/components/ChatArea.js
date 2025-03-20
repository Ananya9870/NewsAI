import React, { useState } from "react";

function ChatArea() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);
    setInput("");
    const botResponse = {
      text: `Summary for "${input}": Here's what I found...`,
      sender: "bot",
    };
    setTimeout(() => setMessages((prev) => [...prev, botResponse]), 500);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className="flex flex-col flex-1">
      <div className="flex-1 overflow-y-auto bg-[#343541] p-4 space-y-4">
        {messages.length === 0 ? (
          <div
            className="text-center text-[# [continues as before]
          text-[#d1d5db] opacity-50 mt-20"
          >
            Start asking about the news!
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg ${
                msg.sender === "user"
                  ? "bg-[#40414f] text-[#d1d5db] ml-auto max-w-[80%]"
                  : "bg-[#444654] text-[#ececf1] max-w-[80%]"
              }`}
            >
              {msg.text}
            </div>
          ))
        )}
      </div>
      <div className="mt-4 flex items-center p-2 bg-[#40414f] rounded-lg shadow-md">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          className="flex-1 p-2 bg-transparent text-[#d1d5db] placeholder-[#8e8ea0] focus:outline-none"
          placeholder="Type your prompt..."
        />
        <button
          onClick={handleSend}
          className="p-2 bg-[#10a37f] text-white rounded-md hover:bg-[#0d8c6b] transition-colors"
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
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}

export default ChatArea;
