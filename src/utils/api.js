// API service for communication with the backend
import config from './config';

const API_BASE_URL = config.api.baseUrl;

// Generate a random session ID if one doesn't exist
const getSessionId = () => {
  if (!localStorage.getItem('sessionId')) {
    const randomId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('sessionId', randomId);
  }
  return localStorage.getItem('sessionId');
};

// Send a chat message to the backend
export const sendChatMessage = async (message, language = 'en', pincode = null) => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: getSessionId(),
        language,
        pincode,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
};

// Look up location from pincode
export const lookupPincode = async (pincode) => {
  try {
    const response = await fetch(`${API_BASE_URL}/pincode`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pincode,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error looking up pincode:', error);
    throw error;
  }
};

// Get news for a specific location
export const getLocationNews = async (location, count = 5, language = 'en') => {
  try {
    const response = await fetch(`${API_BASE_URL}/news/${encodeURIComponent(location)}?count=${count}&language=${language}`);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching location news:', error);
    throw error;
  }
};

// Translate text to a different language
export const translateText = async (text, targetLanguage) => {
  try {
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        target_language: targetLanguage,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error translating text:', error);
    throw error;
  }
};

// Get text-to-speech audio for a given text
export const getTextToSpeech = async (text, lang = 'en') => {
  try {
    const response = await fetch(`${API_BASE_URL}/text-to-speech`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        lang,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return response;
  } catch (error) {
    console.error('Error getting text-to-speech:', error);
    throw error;
  }
};

// Get list of supported languages
export const getSupportedLanguages = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/languages`);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching supported languages:', error);
    throw error;
  }
};

export default {
  sendChatMessage,
  lookupPincode,
  getLocationNews,
  translateText,
  getTextToSpeech,
  getSupportedLanguages,
}; 