import React, { useEffect, useState } from "react";
import LanguageSelector from "./LanguageSelector";
import LocationSelector from "./LocationSelector";
import { useUserPreferences } from "../context/UserPreferences";
import { getNewsByLocation, indexDatabase } from "../utils/database";

function NewsDashboard() {
  const [news, setNews] = useState([]);
  const [isLoading, setIsLoading] = useState(true); // Start with loading true
  const { language, location } = useUserPreferences();

  // Initialize database on component mount
  useEffect(() => {
    try {
      indexDatabase(); // Just index the database without storing stats
    } catch (error) {
      console.error("Error initializing database:", error);
    }
  }, []); // Empty dependency array: runs once on mount

  // Fetch news when language or location changes
  useEffect(() => {
    let isMounted = true; // Flag to track component mount status

    const fetchNewsData = async () => {
      if (!location) {
        if (isMounted) {
          setNews([]); // Clear news if no location
          setIsLoading(false); // Not loading anything if no location
        }
        return;
      }
      
      if (isMounted) {
        setIsLoading(true);
      }
      
      try {
        // Simulate a delay to show loading state
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Get news for the selected location
        // Assuming getNewsByLocation is synchronous; if it were async, add await
        const locationNews = getNewsByLocation(location.name, language);
        
        if (isMounted) {
          if (locationNews && locationNews.length > 0) {
            setNews(locationNews);
          } else {
            // Fallback - if no news for this location, just use mock data
            const locationName = location.name || "your area";
            const mockNews = [
              {
                id: 1,
                title: `Local Event Today in ${locationName}`, // Made title more dynamic for clarity
                summary: `A festival is happening in ${locationName}.`,
                location: locationName,
                category: "entertainment",
                date: new Date().toISOString().slice(0, 10)
              },
              {
                id: 2,
                title: `Weather Update for ${locationName}`, // Made title more dynamic for clarity
                summary: `Sunny with a high of 75°F in ${locationName}.`,
                location: locationName,
                category: "weather",
                date: new Date().toISOString().slice(0, 10)
              },
            ];
            setNews(mockNews);
          }
        }
      } catch (error) {
        console.error("Failed to fetch news:", error);
        if (isMounted) {
          setNews([]); // Clear news on error
          // Optionally, set an error message to display to the user
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };
    
    fetchNewsData();

    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [language, location]); // Dependencies: language, location

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">Language</h3>
        <LanguageSelector />
      </div>
      
      <LocationSelector />
      
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-white mb-2">Local News</h3>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#3b82f6]"></div>
          </div>
        ) : news.length > 0 ? (
          <div className="space-y-4 max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
            {news.map((item) => (
              <div
                key={item.id}
                className="p-4 bg-[#343541] rounded-lg shadow-md border border-[#444654] w-full"
              >
                <h3 className="text-lg font-semibold text-white uppercase">
                  {item.title}
                </h3>
                <p className="text-[#d1d5db]">{item.summary}</p>
                {item.category && (
                  <span className="inline-block mt-2 px-2 py-1 bg-[#444654] text-xs text-[#d1d5db] rounded">
                    {item.category}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[#d1d5db]">No news available for your location or an error occurred.</p>
        )}
      </div>
    </div>
  );
}

export default NewsDashboard;