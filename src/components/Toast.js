import React, { useState, useEffect } from 'react';

const Toast = ({ message, duration = 3000, onClose }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(() => {
        if (onClose) onClose();
      }, 300); // Wait for fade out animation
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  return (
    <div
      className={`fixed bottom-16 right-4 bg-[#1e40af] text-white px-4 py-2 rounded-md shadow-lg transition-opacity duration-300 z-50 
        ${isVisible ? 'opacity-100' : 'opacity-0'}`}
    >
      {message}
    </div>
  );
};

export default Toast; 