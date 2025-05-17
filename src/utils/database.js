// Database utility functions for the news summarizer app
// In a real application, this would interact with an actual database
// For now, it simulates database operations with localStorage

// Simulated news database
const newsData = {
  en: [
    {
      id: 1,
      title: "Local Festival",
      summary: "Annual cultural festival downtown this weekend.",
      location: "New York",
      category: "entertainment",
      date: "2023-06-15"
    },
    {
      id: 2,
      title: "New Infrastructure Project",
      summary: "City announces new road renovation project.",
      location: "London",
      category: "politics",
      date: "2023-06-14"
    },
    {
      id: 3,
      title: "Tech Conference",
      summary: "Major tech companies gathering for annual conference.",
      location: "Tokyo",
      category: "technology",
      date: "2023-06-16"
    }
  ],
  // Example of news in another language
  es: [
    {
      id: 1,
      title: "Festival Local",
      summary: "Festival cultural anual en el centro este fin de semana.",
      location: "New York",
      category: "entertainment",
      date: "2023-06-15"
    },
    {
      id: 2,
      title: "Nuevo Proyecto de Infraestructura",
      summary: "La ciudad anuncia un nuevo proyecto de renovación de carreteras.",
      location: "London",
      category: "politics",
      date: "2023-06-14"
    }
  ]
};

// Simulated database indexes
let locationIndex = {};
let categoryIndex = {};
let dateIndex = {};

// Function to initialize and index the database
export const indexDatabase = () => {
  console.log("Indexing database...");
  
  // Create indexes
  Object.entries(newsData).forEach(([language, articles]) => {
    // Initialize indexes for this language if they don't exist
    if (!locationIndex[language]) locationIndex[language] = {};
    if (!categoryIndex[language]) categoryIndex[language] = {};
    if (!dateIndex[language]) dateIndex[language] = {};
    
    // Index each article
    articles.forEach(article => {
      // Location index
      if (!locationIndex[language][article.location]) {
        locationIndex[language][article.location] = [];
      }
      locationIndex[language][article.location].push(article.id);
      
      // Category index
      if (!categoryIndex[language][article.category]) {
        categoryIndex[language][article.category] = [];
      }
      categoryIndex[language][article.category].push(article.id);
      
      // Date index
      if (!dateIndex[language][article.date]) {
        dateIndex[language][article.date] = [];
      }
      dateIndex[language][article.date].push(article.id);
    });
  });
  
  console.log("Database indexed successfully!");
  
  // Save indexes to localStorage (in a real app, this would be in a database)
  saveIndexes();
  
  return {
    totalArticles: Object.values(newsData).reduce((total, articles) => total + articles.length, 0),
    languages: Object.keys(newsData).length,
    locations: Object.keys(locationIndex).reduce((total, lang) => total + Object.keys(locationIndex[lang]).length, 0),
    categories: Object.keys(categoryIndex).reduce((total, lang) => total + Object.keys(categoryIndex[lang]).length, 0)
  };
};

// Save indexes to localStorage
const saveIndexes = () => {
  localStorage.setItem('newsLocationIndex', JSON.stringify(locationIndex));
  localStorage.setItem('newsCategoryIndex', JSON.stringify(categoryIndex));
  localStorage.setItem('newsDateIndex', JSON.stringify(dateIndex));
};

// Load indexes from localStorage
const loadIndexes = () => {
  const storedLocationIndex = localStorage.getItem('newsLocationIndex');
  const storedCategoryIndex = localStorage.getItem('newsCategoryIndex');
  const storedDateIndex = localStorage.getItem('newsDateIndex');
  
  if (storedLocationIndex) locationIndex = JSON.parse(storedLocationIndex);
  if (storedCategoryIndex) categoryIndex = JSON.parse(storedCategoryIndex);
  if (storedDateIndex) dateIndex = JSON.parse(storedDateIndex);
};

// Get news by location
export const getNewsByLocation = (location, language = 'en') => {
  loadIndexes();
  
  // If no indexes exist, create them
  if (Object.keys(locationIndex).length === 0) {
    indexDatabase();
  }
  
  // If language is not available, fall back to English
  if (!locationIndex[language]) {
    language = 'en';
  }
  
  // If location is not indexed, return empty array
  if (!locationIndex[language][location]) {
    return [];
  }
  
  // Get article IDs for this location
  const articleIds = locationIndex[language][location];
  
  // Return the actual articles
  return articleIds.map(id => 
    newsData[language].find(article => article.id === id)
  );
};

// Get news by category
export const getNewsByCategory = (category, language = 'en') => {
  loadIndexes();
  
  // If no indexes exist, create them
  if (Object.keys(categoryIndex).length === 0) {
    indexDatabase();
  }
  
  // If language is not available, fall back to English
  if (!categoryIndex[language]) {
    language = 'en';
  }
  
  // If category is not indexed, return empty array
  if (!categoryIndex[language][category]) {
    return [];
  }
  
  // Get article IDs for this category
  const articleIds = categoryIndex[language][category];
  
  // Return the actual articles
  return articleIds.map(id => 
    newsData[language].find(article => article.id === id)
  );
};

// Get news by date
export const getNewsByDate = (date, language = 'en') => {
  loadIndexes();
  
  // If no indexes exist, create them
  if (Object.keys(dateIndex).length === 0) {
    indexDatabase();
  }
  
  // If language is not available, fall back to English
  if (!dateIndex[language]) {
    language = 'en';
  }
  
  // If date is not indexed, return empty array
  if (!dateIndex[language][date]) {
    return [];
  }
  
  // Get article IDs for this date
  const articleIds = dateIndex[language][date];
  
  // Return the actual articles
  return articleIds.map(id => 
    newsData[language].find(article => article.id === id)
  );
};

// Search news articles (simple implementation)
export const searchNews = (query, language = 'en') => {
  // If language is not available, fall back to English
  if (!newsData[language]) {
    language = 'en';
  }
  
  // Simple search implementation
  const lowerQuery = query.toLowerCase();
  return newsData[language].filter(article => 
    article.title.toLowerCase().includes(lowerQuery) || 
    article.summary.toLowerCase().includes(lowerQuery)
  );
};

// Initialize the database on load
loadIndexes();
if (Object.keys(locationIndex).length === 0) {
  indexDatabase();
} 