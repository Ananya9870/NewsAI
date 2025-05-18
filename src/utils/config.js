// Application configuration

const config = {
  // API configuration
  api: {
    baseUrl: 'http://localhost:8000/api', // Change this to your backend URL
    timeout: 30000, // API request timeout in milliseconds
  },
  
  // Feature flags
  features: {
    textToSpeech: true,
    locationDetection: true,
    saveHistory: true,
  },
};

export default config; 