import React, { useState } from 'react';
import { getUserLocation, defaultLocations, getCityFromCoords } from '../utils/location';
import { useUserPreferences } from '../context/UserPreferences';
import { lookupPincode } from '../utils/api';

function LocationSelector() {
  const { location, updateLocation } = useUserPreferences();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [pincode, setPincode] = useState('');
  const [isPincodeLoading, setIsPincodeLoading] = useState(false);

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
  
  const handlePincodeSubmit = async (e) => {
    e.preventDefault();
    if (!pincode || pincode.length < 4) {
      setError('Please enter a valid postal/zip code');
      return;
    }
    
    setIsPincodeLoading(true);
    setError(null);
    
    try {
      const response = await lookupPincode(pincode);
      if (response && response.location) {
        // Create a location object with the pincode result
        const locationObj = {
          name: response.location,
          pincode: response.pincode,
          // Add dummy coordinates to ensure compatibility
          latitude: null,
          longitude: null,
          source: 'pincode'
        };
        updateLocation(locationObj);
        setIsExpanded(false);
      } else {
        setError('Could not find location for this postal/zip code');
      }
    } catch (err) {
      setError('Error looking up postal/zip code. Please try again or select from the list.');
    } finally {
      setIsPincodeLoading(false);
    }
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
            {location && location.latitude && location.longitude && (
              <p className="text-sm text-[#8e8ea0]">
                {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
              </p>
            )}
            {location && location.pincode && (
              <p className="text-sm text-[#8e8ea0]">
                Pincode: {location.pincode}
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
            
            {/* Pincode Input */}
            <div className="mb-3">
              <form onSubmit={handlePincodeSubmit} className="flex gap-2">
                <input
                  type="text"
                  value={pincode}
                  onChange={(e) => setPincode(e.target.value)}
                  placeholder="Enter postal/zip code"
                  className="flex-1 p-2 bg-[#242731] text-white rounded-md focus:outline-none focus:ring-1 focus:ring-[#3b82f6]"
                />
                <button
                  type="submit"
                  disabled={isPincodeLoading}
                  className="p-2 bg-[#3b82f6] text-white rounded-md hover:bg-[#2563eb] transition-colors"
                >
                  {isPincodeLoading ? (
                    <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                  ) : (
                    'Search'
                  )}
                </button>
              </form>
            </div>
            
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