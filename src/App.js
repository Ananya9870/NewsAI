import React, { useState } from "react";
import NewsDashboard from "./components/NewsDashboard";
import ChatArea from "./components/ChatArea";
import "./App.css";

function App() {
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#343541] flex flex-col relative">
      {/* Hamburger Button */}
      <button
        onClick={() => setIsDashboardOpen(!isDashboardOpen)}
        className="fixed top-4 left-4 z-20 p-2 bg-[#40414f] rounded-md text-[#d1d5db] hover:bg-[#4a4b5e] transition-colors"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>

      {/* Dashboard Panel */}
      <div
        className={`fixed top-0 left-0 h-full w-80 bg-gray-100 text-gray-800 transform transition-transform duration-300 z-10 ${
          isDashboardOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-4">Local News Dashboard</h1>
          <NewsDashboard />
        </div>
      </div>

      {/* Chat Area - Full Screen */}
      <div className="flex-1 p-6 flex flex-col">
        <ChatArea />
      </div>
    </div>
  );
}

export default App;
