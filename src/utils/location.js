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

// Default locations for fallback - expanded with more specific city/state names
export const defaultLocations = [
  { name: 'New York, NY', latitude: 40.7128, longitude: -74.0060 },
  { name: 'London, UK', latitude: 51.5074, longitude: -0.1278 },
  { name: 'Tokyo, Japan', latitude: 35.6762, longitude: 139.6503 },
  { name: 'Sydney, NSW', latitude: -33.8688, longitude: 151.2093 },
  { name: 'Mumbai, India', latitude: 19.0760, longitude: 72.8777 },
  { name: 'Paris, France', latitude: 48.8566, longitude: 2.3522 },
  { name: 'Berlin, Germany', latitude: 52.5200, longitude: 13.4050 },
  { name: 'Toronto, Canada', latitude: 43.6532, longitude: -79.3832 },
  { name: 'Singapore', latitude: 1.3521, longitude: 103.8198 },
  { name: 'Cape Town, South Africa', latitude: -33.9249, longitude: 18.4241 },
  { name: 'Dubai, UAE', latitude: 25.2048, longitude: 55.2708 },
  { name: 'Mexico City, Mexico', latitude: 19.4326, longitude: -99.1332 },
  { name: 'Rio de Janeiro, Brazil', latitude: -22.9068, longitude: -43.1729 },
  { name: 'Moscow, Russia', latitude: 55.7558, longitude: 37.6173 },
  { name: 'Beijing, China', latitude: 39.9042, longitude: 116.4074 },
];

// Find the closest default location based on coordinates
export const findNearestCity = (latitude, longitude) => {
  if (!latitude || !longitude) return null;
  
  // Calculate distances to default locations
  const distances = defaultLocations.map(location => {
    const latDiff = location.latitude - latitude;
    const lngDiff = location.longitude - longitude;
    // Simple Euclidean distance calculation
    return {
      ...location,
      distance: Math.sqrt(latDiff * latDiff + lngDiff * lngDiff)
    };
  });
  
  // Sort by distance
  distances.sort((a, b) => a.distance - b.distance);
  
  // Return the closest city name, regardless of distance
  return distances[0];
};

// Get city name from coordinates using reverse geocoding API
// In a real app, this would call a geocoding service
export const getCityFromCoords = async (latitude, longitude) => {
  try {
    // In a real app, this would call a geocoding API like Google Maps, MapBox, etc.
    // For example: const response = await fetch(`https://api.example.com/geocode?lat=${latitude}&lng=${longitude}`);
    
    // For this demo, we'll just use our findNearestCity function
    const nearestLocation = findNearestCity(latitude, longitude);
    
    if (nearestLocation) {
      return nearestLocation;
    } else {
      // If we can't find a nearby city, use coordinates as fallback
      return {
        name: `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
        latitude,
        longitude
      };
    }
  } catch (error) {
    console.error('Error finding city name:', error);
    return {
      name: `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
      latitude,
      longitude
    };
  }
};