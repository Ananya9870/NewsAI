import React, { useEffect, useState } from "react";
import LanguageSelector from "./LanguageSelector";
import LocationSelector from "./LocationSelector";
import { useUserPreferences } from "../context/UserPreferences";
import { getNewsByLocation, indexDatabase } from "../utils/database";

function NewsDashboard() {
  const [news, setNews] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [indexStats, setIndexStats] = useState(null);
  const { language, location } = useUserPreferences();

  // Initialize database on component mount
  useEffect(() => {
    const initializeDatabase = async () => {
      setIsLoading(true);
      const stats = indexDatabase();
      setIndexStats(stats);
      setIsLoading(false);
    };

    initializeDatabase();
  }, []);

  // Fetch news when language or location changes
  useEffect(() => {
    const fetchNews = async () => {
      if (!location) return;
      
      setIsLoading(true);
      // Simulate a delay to show loading state
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Get news for the selected location
      const locationNews = getNewsByLocation(location.name, language);
      
      if (locationNews.length > 0) {
        setNews(locationNews);
      } else {
        // Fallback - if no news for this location, just use mock data
        const mockNews = [
          {
            id: 1,
            title: "Local Event Today",
            summary: `A festival is happening in ${location.name}.`,
            location: location.name,
            category: "entertainment",
            date: new Date().toISOString().slice(0, 10)
          },
          {
            id: 2,
            title: "Weather Update",
            summary: `Sunny with a high of 75°F in ${location.name}.`,
            location: location.name,
            category: "weather",
            date: new Date().toISOString().slice(0, 10)
          },
        ];
        setNews(mockNews);
      }
      
      setIsLoading(false);
    };
    
    fetchNews();
  }, [language, location]);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">Language</h3>
        <LanguageSelector />
      </div>
      
      <LocationSelector />
      
      {indexStats && (
        <div className="mt-4 p-3 bg-[#343541] rounded-lg text-sm text-[#8e8ea0]">
          <p>Database indexed with {indexStats.totalArticles} articles</p>
          <p>Languages: {indexStats.languages}</p>
          <p>Locations: {indexStats.locations}</p>
        </div>
      )}
      
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-white mb-2">Local News</h3>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#3b82f6]"></div>
          </div>
        ) : news.length > 0 ? (
          <div className="space-y-4">
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
          <p className="text-[#d1d5db]">No news available for your location.</p>
        )}
      </div>
    </div>
  );
}

export default NewsDashboard;
