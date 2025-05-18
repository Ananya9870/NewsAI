import React, { useEffect, useState } from "react";
import LanguageSelector from "./LanguageSelector";
import LocationSelector from "./LocationSelector";
import { useUserPreferences } from "../context/UserPreferences";
import { getLocationNews } from "../utils/api";

function NewsDashboard() {
  const [news, setNews] = useState([]);
  const [isLoading, setIsLoading] = useState(true); // Start with loading true
  const { language, location } = useUserPreferences();

  // Fetch news when language or location changes
  useEffect(() => {
    let isMounted = true; // Flag to track component mount status

    const fetchNewsData = async () => {
      if (!location || !location.name) {
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
        // Get location name
        const locationName = location.name;
        
        // Get news from the API
        const response = await getLocationNews(locationName, 5, language);
        
        if (isMounted) {
          if (response && response.news) {
            // Parse the news from the response
            // The API might return news as a string, so we might need to parse it
            let parsedNews;
            
            if (typeof response.news === 'string') {
              // Try to extract news items from the text response
              const newsItems = response.news.split('\n\n').filter(item => item.trim());
              parsedNews = newsItems.map((item, index) => {
                const title = item.split('\n')[0].replace(/^[-*]\s*/, '');
                const summary = item.split('\n').slice(1).join('\n');
                
                return {
                  id: index + 1,
                  title: title || `News ${index + 1}`,
                  summary: summary || item,
                  location: locationName,
                  category: "news",
                  date: new Date().toISOString().slice(0, 10)
                };
              });
            } else if (Array.isArray(response.news)) {
              // Use the array directly
              parsedNews = response.news;
            } else {
              // Fallback for unexpected format
              parsedNews = [];
            }
            
            setNews(parsedNews);
          } else {
            // Fallback - if no news for this location, just use mock data
            const locationName = location.name || "your area";
            const mockNews = [
              {
                id: 1,
                title: `Local News from ${locationName}`,
                summary: `No specific news found for ${locationName}. Ask in the chat for more information.`,
                location: locationName,
                category: "information",
                date: new Date().toISOString().slice(0, 10)
              }
            ];
            setNews(mockNews);
          }
        }
      } catch (error) {
        console.error("Failed to fetch news:", error);
        if (isMounted) {
          // Fallback on error
          const errorNews = [{
            id: 1,
            title: "Unable to Fetch News",
            summary: "There was an error fetching news. Please try again later or ask in the chat for news updates.",
            location: location.name,
            category: "error",
            date: new Date().toISOString().slice(0, 10)
          }];
          setNews(errorNews);
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

  // Function to extract URLs from text
  const extractLinks = (text) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const matches = text.match(urlRegex);
    
    if (matches) {
      return matches[0]; // Return first link
    }
    
    return null;
  };

  // Function to get color based on category
  const getCategoryColor = (category) => {
    const categoryColors = {
      "news": "bg-[#3b82f6]",
      "entertainment": "bg-[#8b5cf6]",
      "sports": "bg-[#10b981]",
      "politics": "bg-[#ef4444]",
      "technology": "bg-[#6366f1]",
      "health": "bg-[#14b8a6]",
      "business": "bg-[#f59e0b]",
      "science": "bg-[#6366f1]",
      "information": "bg-[#6b7280]",
      "error": "bg-[#ef4444]"
    };
    
    return categoryColors[category?.toLowerCase()] || "bg-[#6b7280]";
  };

  // Format date in a more readable way
  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString(language === 'en' ? 'en-US' : language, { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      });
    } catch (e) {
      return dateString;
    }
  };

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
            {news.map((item) => {
              const link = extractLinks(item.summary);
              
              return (
                <div
                  key={item.id}
                  className="p-4 bg-[#343541] rounded-lg shadow-md border border-[#444654] w-full hover:border-[#3b82f6] transition-colors"
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-lg font-semibold text-white break-words pr-2">
                      {item.title}
                    </h3>
                    <div className="flex-shrink-0">
                      <span className={`inline-block px-2 py-1 text-xs text-white rounded ${getCategoryColor(item.category)}`}>
                        {item.category}
                      </span>
                    </div>
                  </div>
                  
                  <p className="text-[#d1d5db] mb-3 break-words">
                    {item.summary.replace(link, '')}
                  </p>
                  
                  <div className="flex justify-between items-center text-xs text-[#8e8ea0]">
                    <span>{formatDate(item.date)}</span>
                    {link && (
                      <a 
                        href={link} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="bg-[#3b82f6] text-white px-2 py-1 rounded-md hover:bg-[#2563eb] transition-colors text-xs"
                      >
                        Read More
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-6 bg-[#343541] rounded-lg shadow-md border border-[#444654] text-center">
            <p className="text-[#d1d5db]">No news available for your location.</p>
            <p className="text-[#8e8ea0] mt-2">Try changing your location or asking in chat.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default NewsDashboard;