import React, { createContext, useState, useContext, useEffect } from 'react';
import { getSavedLocation, saveLocation } from '../utils/location';

// Create the context
const UserPreferencesContext = createContext();

// Custom hook to use the context
export const useUserPreferences = () => useContext(UserPreferencesContext);

// Provider component
export const UserPreferencesProvider = ({ children }) => {
  const [language, setLanguage] = useState('en');
  const [location, setLocation] = useState(null);
  const [isFirstVisit, setIsFirstVisit] = useState(false);

  // Load saved preferences on initial mount
  useEffect(() => {
    // Load saved language
    const savedLanguage = localStorage.getItem('userLanguage');
    if (savedLanguage) {
      setLanguage(savedLanguage);
    }

    // Load saved location
    const savedLocation = getSavedLocation();
    if (savedLocation) {
      setLocation(savedLocation);
    } else {
      // If no location is saved, it might be the first visit
      setIsFirstVisit(true);
    }
  }, []);

  // Update language preference
  const updateLanguage = (newLanguage) => {
    setLanguage(newLanguage);
    localStorage.setItem('userLanguage', newLanguage);
  };

  // Update location preference
  const updateLocation = (newLocation) => {
    setLocation(newLocation);
    saveLocation(newLocation);
    setIsFirstVisit(false);
  };

  // Skip location setup
  const skipLocationSetup = () => {
    setIsFirstVisit(false);
  };

  // The value to be provided to consumers
  const value = {
    language,
    updateLanguage,
    location,
    updateLocation,
    isFirstVisit,
    skipLocationSetup
  };

  return (
    <UserPreferencesContext.Provider value={value}>
      {children}
    </UserPreferencesContext.Provider>
  );
};

export default UserPreferencesContext; 