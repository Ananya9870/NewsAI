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
          <div
            key={item.id}
            className="p-4 bg-[#343541] rounded-lg shadow-md border border-[#444654]"
          >
            <h3 className="text-lg font-semibold text-white uppercase">
              {item.title}
            </h3>
            <p className="text-[#d1d5db]">{item.summary}</p>
          </div>
        ))
      ) : (
        <p className="text-[#d1d5db]">Loading news...</p>
      )}
    </div>
  );
}

export default NewsDashboard;
