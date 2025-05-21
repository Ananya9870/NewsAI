import React, { useState, useRef, useEffect } from 'react';

function AudioPlayer({ audioSrc, onClose }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const audioRef = useRef(null);

  useEffect(() => {
    const audioElement = audioRef.current;
    
    if (audioElement) {
      const handlePlay = () => setIsPlaying(true);
      const handlePause = () => setIsPlaying(false);
      const handleEnded = () => {
        setIsPlaying(false);
        setProgress(0);
      };
      const handleTimeUpdate = () => {
        const value = (audioElement.currentTime / audioElement.duration) * 100;
        setProgress(value);
      };
      
      // Add event listeners
      audioElement.addEventListener('play', handlePlay);
      audioElement.addEventListener('pause', handlePause);
      audioElement.addEventListener('ended', handleEnded);
      audioElement.addEventListener('timeupdate', handleTimeUpdate);
      
      // Clean up
      return () => {
        audioElement.removeEventListener('play', handlePlay);
        audioElement.removeEventListener('pause', handlePause);
        audioElement.removeEventListener('ended', handleEnded);
        audioElement.removeEventListener('timeupdate', handleTimeUpdate);
      };
    }
  }, []);

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
    }
  };

  return (
    <div className="fixed bottom-4 right-4 bg-[#2a2d37] text-white p-3 rounded-lg shadow-lg z-50 max-w-xs w-full flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-sm font-medium">Text-to-Speech</h4>
        <button
          onClick={onClose}
          className="text-[#8e8ea0] hover:text-white"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
      
      <div className="w-full bg-[#40414f] rounded-full h-2 mb-2">
        <div
          className="bg-[#3b82f6] h-2 rounded-full"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      
      <div className="flex justify-center">
        <button
          onClick={togglePlayPause}
          className="p-2 bg-[#3b82f6] rounded-full hover:bg-[#2563eb] transition-colors"
        >
          {isPlaying ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
          )}
        </button>
      </div>
      
      <audio ref={audioRef} src={audioSrc} preload="metadata" />
    </div>
  );
}

export default AudioPlayer; 