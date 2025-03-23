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
    setMessages((prev) => [...prev, botResponse]);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className="flex flex-col flex-1 w-full max-w-3xl">
      <div className="flex-1 overflow-y-auto bg-[#1e2129] p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-[#e5e7eb] opacity-70 mt-20">
            Start asking about the news!
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg, index) => (
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
            ))}
          </div>
        )}
      </div>
      <div className="mt-4 flex items-center p-2 bg-[#40414f] rounded-lg shadow-md w-full max-w-3xl">
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
    </div>
  );
}

export default ChatArea;
