import React, { useState } from 'react';
import { getUserLocation, defaultLocations, getCityFromCoords } from '../utils/location';
import { useUserPreferences } from '../context/UserPreferences';

function LocationSelector() {
  const { location, updateLocation } = useUserPreferences();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleAutoDetect = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const coords = await getUserLocation();
      // Try to get a city name from coordinates
      try {
        const locationObj = await getCityFromCoords(coords.latitude, coords.longitude);
        updateLocation(locationObj);
      } catch (e) {
        updateLocation({
          name: 'Current Location',
          ...coords
        });
      }
    } catch (err) {
      setError('Could not detect your location. Please select from the list below.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLocationSelect = (selectedLocation) => {
    updateLocation(selectedLocation);
    setIsExpanded(false);
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-white mb-2">Location</h3>
      
      <div className="p-3 bg-[#343541] rounded-lg shadow-md border border-[#444654]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white font-medium">
              {location?.name || 'Location not set'}
            </p>
            {location && (
              <p className="text-sm text-[#8e8ea0]">
                {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
              </p>
            )}
          </div>
          <button
            onClick={toggleExpanded}
            className="p-1 text-[#8e8ea0] hover:text-white"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className={`h-5 w-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d={isExpanded ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} 
              />
            </svg>
          </button>
        </div>
        
        {isExpanded && (
          <div className="mt-3 border-t border-[#444654] pt-3">
            {error && (
              <div className="p-2 mb-3 bg-red-500 bg-opacity-20 border border-red-500 rounded text-white text-sm">
                {error}
              </div>
            )}
            
            <button
              onClick={handleAutoDetect}
              disabled={isLoading}
              className="w-full p-2 mb-3 bg-[#3b82f6] text-white rounded hover:bg-[#2563eb] transition-colors flex items-center justify-center"
            >
              {isLoading ? (
                <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                </svg>
              )}
              {isLoading ? 'Detecting...' : 'Use current location'}
            </button>
            
            <div className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar">
              {defaultLocations.map((loc) => (
                <div
                  key={loc.name}
                  onClick={() => handleLocationSelect(loc)}
                  className={`p-2 rounded cursor-pointer transition-colors ${
                    location?.name === loc.name
                      ? 'bg-[#3b82f6] text-white'
                      : 'bg-[#444654] text-[#d1d5db] hover:bg-[#535567]'
                  }`}
                >
                  {loc.name}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default LocationSelector; 