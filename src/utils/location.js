// Location handling utility functions

// Get the user's location from the browser's geolocation API
export const getUserLocation = () => {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported by your browser'));
      return;
    }
    
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        resolve({ latitude, longitude });
      },
      (error) => {
        reject(error);
      }
    );
  });
};

// Save location to localStorage
export const saveLocation = (location) => {
  localStorage.setItem('userLocation', JSON.stringify(location));
};

// Get location from localStorage
export const getSavedLocation = () => {
  const location = localStorage.getItem('userLocation');
  return location ? JSON.parse(location) : null;
};

// Default locations for fallback
export const defaultLocations = [
  { name: 'New York', latitude: 40.7128, longitude: -74.0060 },
  { name: 'London', latitude: 51.5074, longitude: -0.1278 },
  { name: 'Tokyo', latitude: 35.6762, longitude: 139.6503 },
  { name: 'Sydney', latitude: -33.8688, longitude: 151.2093 },
  { name: 'Mumbai', latitude: 19.0760, longitude: 72.8777 },
];

// Get city name from coordinates using reverse geocoding API
// This is a placeholder - you would replace with an actual API call
export const getCityFromCoords = async (latitude, longitude) => {
  // In a real app, you would call a geocoding API here
  // For now, we'll just return a placeholder
  return `City at ${latitude.toFixed(2)}, ${longitude.toFixed(2)}`;
}; 