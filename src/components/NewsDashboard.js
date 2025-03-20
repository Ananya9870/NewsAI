import React, { useEffect, useState } from "react";

function NewsDashboard() {
  const [news, setNews] = useState([]);

  useEffect(() => {
    const fetchNews = async () => {
      const mockNews = [
        {
          id: 1,
          title: "Local Event Today",
          summary: "A festival is happening downtown.",
        },
        {
          id: 2,
          title: "Weather Update",
          summary: "Sunny with a high of 75°F.",
        },
      ];
      setNews(mockNews);
    };
    fetchNews();
  }, []);

  return (
    <div className="space-y-4">
      {news.length > 0 ? (
        news.map((item) => (
          <div key={item.id} className="p-4 bg-white rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-800">
              {item.title}
            </h3>
            <p className="text-gray-600">{item.summary}</p>
          </div>
        ))
      ) : (
        <p className="text-gray-500">Loading news...</p>
      )}
    </div>
  );
}

export default NewsDashboard;
