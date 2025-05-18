import React, { useState, useRef, useEffect } from 'react';
import { getLanguageOptions } from '../utils/languages';
import { useUserPreferences } from '../context/UserPreferences';
import { getSupportedLanguages } from '../utils/api';

function LanguageSelector() {
  const { language, updateLanguage } = useUserPreferences();
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState([]);
  const [languageOptions, setLanguageOptions] = useState(getLanguageOptions());
  const [isLoading, setIsLoading] = useState(false);
  const dropdownRef = useRef(null);
  const searchInputRef = useRef(null);
  
  const selectedLanguage = languageOptions.find(option => option.value === language);

  // Fetch supported languages from the backend
  useEffect(() => {
    const fetchLanguages = async () => {
      setIsLoading(true);
      try {
        const response = await getSupportedLanguages();
        if (response && response.languages) {
          // Convert the languages object to our expected format
          const options = Object.entries(response.languages).map(([code, name]) => ({
            value: code,
            label: name
          }));
          setLanguageOptions(options);
        }
      } catch (error) {
        console.error("Error fetching supported languages:", error);
        // Fallback to local languages if API fails
        setLanguageOptions(getLanguageOptions());
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchLanguages();
  }, []);

  // Handle outside click to close dropdown
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Filter options based on search term
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredOptions(languageOptions);
    } else {
      const filtered = languageOptions.filter(option => 
        option.label.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredOptions(filtered);
    }
  }, [searchTerm, languageOptions]);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setSearchTerm('');
      setFilteredOptions(languageOptions);
    }
  };

  const handleSelectLanguage = (selectedOption) => {
    updateLanguage(selectedOption.value);
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={toggleDropdown}
        className="flex items-center justify-between w-full p-2 bg-[#343541] text-white rounded-md hover:bg-[#444654] transition-colors"
        disabled={isLoading}
      >
        {isLoading ? (
          <span className="flex items-center">
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span>
            Loading languages...
          </span>
        ) : (
          <span>{selectedLanguage?.label || 'Select Language'}</span>
        )}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-10 mt-1 w-full bg-[#2a2d37] border border-[#444654] rounded-md shadow-lg">
          <div className="p-2">
            <input
              ref={searchInputRef}
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search language..."
              className="w-full p-2 bg-[#343541] text-white rounded-md focus:outline-none focus:ring-1 focus:ring-[#3b82f6]"
            />
          </div>
          <div className="max-h-60 overflow-y-auto custom-scrollbar">
            {isLoading ? (
              <div className="p-4 text-center text-[#d1d5db]">
                <span className="inline-block w-4 h-4 border-2 border-[#d1d5db] border-t-transparent rounded-full animate-spin mr-2"></span>
                Loading languages...
              </div>
            ) : filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <div
                  key={option.value}
                  onClick={() => handleSelectLanguage(option)}
                  className={`p-2 cursor-pointer hover:bg-[#444654] ${
                    option.value === language ? 'bg-[#3b82f6] text-white' : 'text-[#d1d5db]'
                  }`}
                >
                  {option.label}
                </div>
              ))
            ) : (
              <div className="p-2 text-[#8e8ea0]">No languages found</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default LanguageSelector; 