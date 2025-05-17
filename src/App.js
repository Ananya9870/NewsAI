import React, { useState, useEffect, useRef } from "react";
import { gsap } from "gsap";
import NewsDashboard from "./components/NewsDashboard";
import ChatArea from "./components/ChatArea";
import LocationPrompt from "./components/LocationPrompt";
import { UserPreferencesProvider, useUserPreferences } from "./context/UserPreferences";
import "./App.css";

// Main App content separated to use the context
function AppContent() {
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const { isFirstVisit } = useUserPreferences();
  const dashboardRef = useRef(null);
  const contentRef = useRef(null);

  useEffect(() => {
    if (dashboardRef.current) {
      gsap.to(dashboardRef.current, {
        width: isDashboardOpen ? "20rem" : "0rem",
        duration: 0.2,
        ease: "expo.inOut'",
        onStart: () => {
          if (!isDashboardOpen) {
            dashboardRef.current.style.overflow = "hidden";
            dashboardRef.current.style.visibility = "hidden";
          }
        },
        onComplete: () => {
          if (isDashboardOpen) {
            dashboardRef.current.style.overflow = "auto";
            dashboardRef.current.style.visibility = "visible";
          }
        },
      });

      // Fade animation for the dashboard content
      gsap.to(contentRef.current, {
        opacity: isDashboardOpen ? 1 : 0,
        duration: 0.3,
        ease: "power2.inOut",
      });
    }
  }, [isDashboardOpen]);

  return (
    <div className="h-screen bg-[#1e2129] flex flex-row overflow-hidden relative">
      {/* Location Prompt on first visit */}
      {isFirstVisit && <LocationPrompt />}

      {/* Hamburger Button */}
      <button
        onClick={() => setIsDashboardOpen(!isDashboardOpen)}
        className="fixed top-4 left-4 z-40 p-2 bg-[#2a2d37] rounded-md text-[#e5e7eb] hover:bg-[#3a3d47] transition-colors"
      >
        <div className={`hamburger ${isDashboardOpen ? "open" : ""}`}>
          <span></span>
          <span></span>
          <span></span>
        </div>
      </button>

      {/* Dashboard */}
      <div
        ref={dashboardRef}
        className="h-full w-80 bg-[#2a2d37] text-[#e5e7eb] flex-shrink-0 overflow-hidden custom-scrollbar"
      >
        <div ref={contentRef} className="p-6 pt-16">
          <h1 className="text-2xl font-bold mb-4">Local News Dashboard</h1>
          <NewsDashboard />
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 h-full flex flex-col p-6">
        <ChatArea />
      </div>

      {/* Inline Styles for Hamburger Animation */}
      <style>{`
        .hamburger {
          width: 24px;
          height: 18px;
          position: relative;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
        }
        .hamburger span {
          width: 100%;
          height: 2px;
          background: #e5e7eb;
          transition: all 0.3s ease;
        }
        .hamburger.open span:nth-child(1) {
          transform: rotate(45deg);
          position: absolute;
          top: 8px;
        }
        .hamburger.open span:nth-child(2) {
          opacity: 0;
        }
        .hamburger.open span:nth-child(3) {
          transform: rotate(-45deg);
          position: absolute;
          top: 8px;
        }
      `}</style>
    </div>
  );
}

// Main App component that provides the context
function App() {
  return (
    <UserPreferencesProvider>
      <AppContent />
    </UserPreferencesProvider>
  );
}

export default App;
