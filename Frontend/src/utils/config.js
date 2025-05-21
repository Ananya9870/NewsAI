// Application configuration

const config = {
  // API configuration
  api: {
    baseUrl: 'https://sujoy0011-newsai-backend.hf.space/api', // Change this to your backend URL
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