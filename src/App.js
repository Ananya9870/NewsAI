import React, { useState, useEffect, useRef } from "react";
import { gsap } from "gsap";
import NewsDashboard from "./components/NewsDashboard";
import ChatArea from "./components/ChatArea";
import "./App.css";

function App() {
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const dashboardRef = useRef(null);

  useEffect(() => {
    if (dashboardRef.current) {
      gsap.to(dashboardRef.current, {
        x: isDashboardOpen ? 0 : "-100%",
        duration: 0.2,
        ease: "power3.inOut",
      });
    }
  }, [isDashboardOpen]);

  return (
    <div className="min-h-screen bg-[#1e2129] flex flex-col relative">
      {/* Hamburger Button */}
      <button
        onClick={() => setIsDashboardOpen(!isDashboardOpen)}
        className="fixed top-4 left-4 z-50 p-2 bg-[#2a2d37] rounded-md text-[#e5e7eb] hover:bg-[#3a3d47] transition-colors"
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

      {/* Dashboard */}
      <div
        ref={dashboardRef}
        className="fixed top-0 left-0 h-full w-80 bg-[#2a2d37] text-[#e5e7eb] transform transition-transform duration-300 z-40"
      >
        <div className="p-6 pt-16">
          {" "}
          {/* pt-16 ensures no overlap with button */}
          <h1 className="text-2xl font-bold mb-4">Local News Dashboard</h1>
          <NewsDashboard />
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col items-center justify-between p-6">
        <ChatArea />
      </div>
    </div>
  );
}

export default App;
