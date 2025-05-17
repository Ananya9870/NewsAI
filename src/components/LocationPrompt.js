import React, { useState } from 'react';
import { getUserLocation, defaultLocations } from '../utils/location';
import { useUserPreferences } from '../context/UserPreferences';

function LocationPrompt() {
  const { updateLocation, skipLocationSetup } = useUserPreferences();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedLocation, setSelectedLocation] = useState(null);

  const handleAutoDetect = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const coords = await getUserLocation();
      updateLocation({
        name: 'Current Location',
        ...coords
      });
    } catch (err) {
      setError('Could not detect your location. Please select from the list or enter manually.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualSelection = (location) => {
    setSelectedLocation(location);
  };

  const handleConfirmSelection = () => {
    if (selectedLocation) {
      updateLocation(selectedLocation);
    }
  };

  const handleSkip = () => {
    skipLocationSetup();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 p-4">
      <div className="bg-[#2a2d37] rounded-lg shadow-lg p-6 max-w-md w-full">
        <h2 className="text-2xl font-bold text-white mb-4">Set Your Location</h2>
        <p className="text-[#d1d5db] mb-6">
          To provide you with local news and information, we need to know your location.
        </p>
        
        {error && (
          <div className="p-3 mb-4 bg-red-500 bg-opacity-20 border border-red-500 rounded-md text-white">
            {error}
          </div>
        )}
        
        <button
          onClick={handleAutoDetect}
          disabled={isLoading}
          className="w-full p-3 mb-4 bg-[#3b82f6] text-white rounded-md hover:bg-[#2563eb] transition-colors flex items-center justify-center"
        >
          {isLoading ? (
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
            </svg>
          )}
          {isLoading ? 'Detecting...' : 'Auto-detect my location'}
        </button>
        
        <div className="my-4">
          <h3 className="text-white font-semibold mb-2">Or select a location:</h3>
          <div className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar">
            {defaultLocations.map((location) => (
              <div
                key={location.name}
                onClick={() => handleManualSelection(location)}
                className={`p-3 rounded-md cursor-pointer transition-colors ${
                  selectedLocation?.name === location.name
                    ? 'bg-[#3b82f6] text-white'
                    : 'bg-[#343541] text-[#d1d5db] hover:bg-[#444654]'
                }`}
              >
                {location.name}
              </div>
            ))}
          </div>
        </div>
        
        <div className="flex space-x-4 mt-6">
          <button
            onClick={handleSkip}
            className="flex-1 p-2 bg-transparent border border-[#4a4e5a] text-[#d1d5db] rounded-md hover:bg-[#343541] transition-colors"
          >
            Skip for now
          </button>
          <button
            onClick={handleConfirmSelection}
            disabled={!selectedLocation}
            className={`flex-1 p-2 rounded-md transition-colors ${
              selectedLocation
                ? 'bg-[#3b82f6] text-white hover:bg-[#2563eb]'
                : 'bg-[#4a4e5a] text-[#8e8ea0] cursor-not-allowed'
            }`}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

export default LocationPrompt; 