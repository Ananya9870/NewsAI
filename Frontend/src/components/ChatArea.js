import React, { useState, useEffect, useRef } from "react";
import { useUserPreferences } from "../context/UserPreferences";
import { languages } from "../utils/languages";
import { sendChatMessage, getTextToSpeech } from "../utils/api";
import AudioPlayer from "./AudioPlayer";
import Toast from "./Toast";
import ReactMarkdown from 'react-markdown';

function ChatArea() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [showAudioPlayer, setShowAudioPlayer] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [toast, setToast] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState(null);
  const { language, location } = useUserPreferences();
  const chatHistoryRef = useRef(null);
  const recognitionRef = useRef(null);
  const [voicesLoaded, setVoicesLoaded] = useState(false);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  // Add a welcome message when component mounts
  useEffect(() => {
    const currentLanguage = languages[language] || "English";
    const locationName = location && typeof location === 'object' ? location.name : '';
    const welcomeMessage = {
      text: `Welcome! I can provide news summaries in ${currentLanguage}${locationName ? ` for ${locationName}` : ''}. How can I help you today?`,
      sender: "bot",
    };
    setMessages([welcomeMessage]);
  }, [language, location]);

  // Initialize speech recognition on mount and reinitialize when language changes
  useEffect(() => {
    // Initialize recognition with error handling
    const initRecognition = () => {
      try {
        // Check if browser supports SpeechRecognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          console.error("Speech recognition not supported in this browser");
          return false;
        }

        // Create new instance
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = false;
        
        // Set language
        recognitionRef.current.lang = language;
        
        // Set event handlers
        recognitionRef.current.onresult = (event) => {
          try {
            const transcript = event.results[0][0].transcript;
            setInput(transcript);
            // Automatically send the message after voice input
            setTimeout(() => {
              handleSend(transcript);
            }, 500);
          } catch (err) {
            console.error("Error processing speech result:", err);
            setToast("Error processing speech. Please try again.");
          }
        };
        
        recognitionRef.current.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
          setToast(`Voice recognition error: ${event.error}. Please try again.`);
        };
        
        recognitionRef.current.onend = () => {
          setIsRecording(false);
        };
        
        return true;
      } catch (error) {
        console.error("Error initializing speech recognition:", error);
        return false;
      }
    };

    initRecognition();
    
    // Clean up on unmount
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (e) {
          console.error("Error aborting recognition:", e);
        }
      }
    };
  }, [language]);

  // Initialize speechSynthesis and load voices
  useEffect(() => {
    if ('speechSynthesis' in window) {
      // Load voices
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
          setVoicesLoaded(true);
          return true;
        }
        return false;
      };

      // Try to load voices immediately
      if (!loadVoices()) {
        // If voices aren't loaded yet, add an event listener
        window.speechSynthesis.onvoiceschanged = () => {
          loadVoices();
        };
      }
    }
  }, []);

  // Add listener for speech synthesis end events
  useEffect(() => {
    if ('speechSynthesis' in window) {
      const handleSpeechEnd = () => {
        if (!window.speechSynthesis.speaking) {
          setIsSpeaking(false);
          setSpeakingMessageId(null);
        }
      };
      
      // Check speaking status periodically
      const intervalId = setInterval(() => {
        if (window.speechSynthesis.speaking) {
          setIsSpeaking(true);
        } else if (isSpeaking) {
          handleSpeechEnd();
        }
      }, 100);
      
      return () => {
        clearInterval(intervalId);
      };
    }
  }, [isSpeaking]);

  const handleSend = async (voiceInput = null) => {
    const messageText = voiceInput || input;
    if (!messageText.trim() || isLoading) return;
    
    const userMessage = { text: messageText, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    
    try {
      // Get pincode from location if available
      const pincode = location && location.pincode ? location.pincode : null;
      
      // Send message to the backend
      const response = await sendChatMessage(messageText, language, pincode);
      
      // Extract any links from the response
      const links = extractLinks(response.response);
      
      // Create bot response from the backend response
      const botResponse = {
        text: response.response,
        sender: "bot",
        originalResponse: response.original_response,
        links: links
      };
      
      setMessages((prev) => [...prev, botResponse]);
      
      // If there's an audio URL in the response, set it
      if (response.audio_url) {
        setAudioUrl(response.audio_url);
        setShowAudioPlayer(true);
      }
    } catch (error) {
      console.error("Error communicating with the backend:", error);
      // Add an error message
      const errorMessage = {
        text: "Sorry, I encountered an error while processing your request. Please try again later.",
        sender: "bot",
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const extractLinks = (text) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const matches = text.match(urlRegex);
    
    if (matches) {
      // Filter out duplicate links
      return [...new Set(matches)];
    }
    
    return [];
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  const handleMicClick = () => {
    // If speech recognition not initialized, try to initialize it
    if (!recognitionRef.current) {
      // Check if browser supports SpeechRecognition
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        setToast("Speech recognition not supported in your browser");
        return;
      }
      
      // Try to initialize again
      try {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = false;
        recognitionRef.current.lang = language;
        
        recognitionRef.current.onresult = (event) => {
          try {
            const transcript = event.results[0][0].transcript;
            setInput(transcript);
            // Automatically send the message after voice input
            setTimeout(() => {
              handleSend(transcript);
            }, 500);
          } catch (err) {
            console.error("Error processing speech result:", err);
            setToast("Error processing speech. Please try again.");
          }
        };
        
        recognitionRef.current.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
          setToast(`Voice recognition error: ${event.error}. Please try again.`);
        };
        
        recognitionRef.current.onend = () => {
          setIsRecording(false);
        };
      } catch (error) {
        console.error("Error initializing speech recognition:", error);
        setToast("Failed to initialize speech recognition");
        return;
      }
    }
    
    if (isRecording) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.error("Error stopping recognition:", e);
      }
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
        setToast("Listening... Speak now");
      } catch (e) {
        console.error("Error starting speech recognition:", e);
        setToast("Error starting speech recognition. Please try again.");
        setIsRecording(false);
      }
    }
  };
  
  const handleTextToSpeech = async (text, messageId) => {
    try {
      // If already speaking, stop it first
      if (isSpeaking) {
        stopSpeaking();
        
        // If clicking the same message that's already speaking, just stop it
        if (messageId === speakingMessageId) {
          return;
        }
      }
      
      // Set the current speaking message
      setSpeakingMessageId(messageId);
      setIsSpeaking(true);
      
      // Check if we should use client-side or server-side TTS
      const shouldUseClientSide = language === 'en' && 'speechSynthesis' in window && voicesLoaded;
      
      if (shouldUseClientSide) {
        // Use client-side speech synthesis for English
        const utterance = new SpeechSynthesisUtterance(text);
        // Get voices
        const voices = window.speechSynthesis.getVoices();
        
        // Try to find a voice for English
        const voice = voices.find(v => v.lang === 'en-US' || v.lang === 'en-GB') || voices[0];
        if (voice) {
          utterance.voice = voice;
        }
        
        utterance.lang = 'en';
        utterance.onend = () => {
          setIsSpeaking(false);
          setSpeakingMessageId(null);
        };
        
        window.speechSynthesis.speak(utterance);
      } else {
        // Always use server-side TTS for non-English or if client-side is not available
        setToast("Loading audio...");
        const response = await getTextToSpeech(text, language);
        
        // Create a blob from the response and create an object URL
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Create an audio element to play the speech
        const audio = new Audio(url);
        
        audio.onended = () => {
          setIsSpeaking(false);
          setSpeakingMessageId(null);
          URL.revokeObjectURL(url);
        };
        
        audio.onerror = () => {
          setIsSpeaking(false);
          setSpeakingMessageId(null);
          setToast("Error playing audio");
          URL.revokeObjectURL(url);
        };
        
        // Play the audio
        audio.play().catch(err => {
          console.error("Error playing audio:", err);
          setIsSpeaking(false);
          setSpeakingMessageId(null);
          setToast("Error playing audio");
        });
      }
    } catch (error) {
      console.error("Error generating speech:", error);
      setIsSpeaking(false);
      setSpeakingMessageId(null);
      setToast("Error generating speech. Please try again.");
    }
  };
  
  const stopSpeaking = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    
    setIsSpeaking(false);
    setSpeakingMessageId(null);
  };

  const copyLinkToClipboard = (link) => {
    navigator.clipboard.writeText(link)
      .then(() => {
        // Show toast notification
        setToast("Link copied to clipboard!");
      })
      .catch(err => {
        console.error('Failed to copy link:', err);
        setToast("Failed to copy link");
      });
  };
  
  const closeAudioPlayer = () => {
    setShowAudioPlayer(false);
    // If there's a URL object, revoke it to free memory
    if (audioUrl && audioUrl.startsWith('blob:')) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(null);
  };

  // Function to render text with markdown support
  const renderMarkdownText = (text) => {
    return (
      <ReactMarkdown
        components={{
          // Override components to maintain styling
          p: ({ node, ...props }) => <p className="mb-2" {...props} />,
          strong: ({ node, ...props }) => <span className="font-bold" {...props} />,
          em: ({ node, ...props }) => <span className="italic" {...props} />,
          h1: ({ node, ...props }) => <h1 className="text-xl font-bold mb-2" {...props} />,
          h2: ({ node, ...props }) => <h2 className="text-lg font-bold mb-2" {...props} />,
          h3: ({ node, ...props }) => <h3 className="text-md font-bold mb-2" {...props} />,
          ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-2" {...props} />,
          ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-2" {...props} />,
          li: ({ node, ...props }) => <li className="mb-1" {...props} />,
          a: ({ node, ...props }) => <a className="text-[#3b82f6] underline" target="_blank" rel="noopener noreferrer" {...props} />
        }}
      >
        {text}
      </ReactMarkdown>
    );
  };

  return (
    <div className="flex flex-col h-full w-full max-w-3xl mx-auto">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-white">NewsAI</h1>
        <p className="text-[#8e8ea0]">
          {location && typeof location === 'object' && location.name 
            ? `Local news for ${location.name}` 
            : 'Set your location for local news'} 
          {language && ` • ${languages[language]}`}
        </p>
      </div>
      
      <div
        ref={chatHistoryRef}
        className="flex-1 overflow-y-auto bg-[#1e2129] p-4 space-y-4 custom-scrollbar"
      >
        {messages.length === 0 ? (
          <div className="text-center text-[#e5e7eb] opacity-70 mt-20">
            Start asking about the news!
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`rounded-lg shadow-md border ${
                msg.sender === "user"
                  ? "bg-[#343541] text-white ml-auto max-w-[80%] border-[#444654]"
                  : `bg-[#343541] text-white max-w-[95%] ${msg.isError ? 'border-red-500' : 'border-[#444654]'} break-words hover:border-[#3b82f6] transition-colors`
              }`}
            >
              <div className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm font-semibold text-[#8e8ea0] mb-2">
                    {msg.sender === "user" ? "You" : "NewsAI Assistant"}
                    {msg.isError && <span className="ml-2 text-red-500">(Error)</span>}
                  </h3>
                  
                  {msg.sender === "bot" && !msg.isError && (
                    <button
                      onClick={() => isSpeaking && speakingMessageId === index ? stopSpeaking() : handleTextToSpeech(msg.text, index)}
                      className={`p-1 text-white transition-colors flex-shrink-0 rounded-md ${
                        isSpeaking && speakingMessageId === index ? 'bg-red-500 opacity-100' : 'opacity-70 hover:opacity-100'
                      }`}
                      title={isSpeaking && speakingMessageId === index ? "Stop speaking" : "Listen"}
                    >
                      {isSpeaking && speakingMessageId === index ? (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 002 0V8a1 1 0 00-1-1zm4 0a1 1 0 00-1 1v4a1 1 0 002 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071a1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243a1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828a1 1 0 010-1.415z" clipRule="evenodd" />
                        </svg>
                      )}
                    </button>
                  )}
                </div>
                
                <div className="text-white break-words">
                  {msg.sender === "bot" && !msg.isError ? (
                    renderMarkdownText(msg.text)
                  ) : (
                    msg.text.split('\n').map((line, i) => (
                      <React.Fragment key={i}>
                        {line}
                        {i < msg.text.split('\n').length - 1 && <br />}
                      </React.Fragment>
                    ))
                  )}
                </div>
                
                {/* Display links as clickable buttons if present */}
                {msg.links && msg.links.length > 0 && (
                  <div className="mt-3 p-2 bg-[#2a2d37] rounded-md">
                    <p className="text-sm text-[#8e8ea0] mb-2">News links:</p>
                    <div className="flex flex-wrap gap-2">
                      {msg.links.map((link, i) => (
                        <div key={i} className="inline-flex">
                          <a 
                            href={link} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-xs bg-[#3b82f6] text-white px-2 py-1 rounded-l-md hover:bg-[#2563eb] transition-colors"
                          >
                            Link {i+1}
                          </a>
                          <button
                            onClick={() => copyLinkToClipboard(link)}
                            className="text-xs bg-[#1e3a8a] text-white px-1 py-1 rounded-r-md hover:bg-[#1e40af] transition-colors"
                            title="Copy link"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                              <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
                            </svg>
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex items-center space-x-2 p-4 bg-[#343541] border border-[#444654] text-white rounded-lg max-w-[80%] shadow-md">
            <div className="animate-pulse flex space-x-2">
              <div className="h-2 w-2 bg-[#3b82f6] rounded-full"></div>
              <div className="h-2 w-2 bg-[#3b82f6] rounded-full"></div>
              <div className="h-2 w-2 bg-[#3b82f6] rounded-full"></div>
            </div>
            <span>Getting the latest news...</span>
          </div>
        )}
      </div>
      <div className="mt-4 flex items-center p-2 bg-[#40414f] rounded-lg shadow-md w-full">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          className="flex-1 p-2 bg-transparent text-white placeholder-[#8e8ea0] focus:outline-none"
          placeholder="Type your prompt..."
          disabled={isLoading}
        />
        <button
          onClick={handleMicClick}
          className={`p-2 text-white rounded-md hover:bg-[#4a4e5a] transition-colors ${isRecording ? 'bg-red-500 hover:bg-red-600' : ''}`}
          disabled={isLoading}
          title={isRecording ? "Stop recording" : "Start voice input"}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
        </button>
        <button
          onClick={() => handleSend()}
          className={`p-2 bg-[#3b82f6] text-white rounded-md transition-colors ${
            isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#2563eb]'
          }`}
          disabled={isLoading}
        >
          Send
        </button>
      </div>

      {/* Audio Player */}
      {showAudioPlayer && audioUrl && (
        <AudioPlayer 
          audioSrc={audioUrl} 
          onClose={closeAudioPlayer} 
        />
      )}

      {/* Toast Notification */}
      {toast && (
        <Toast 
          message={toast} 
          onClose={() => setToast(null)} 
        />
      )}

      {/* Inline Custom Scrollbar Styles */}
      <style>{`
        .custom-scrollbar {
          scrollbar-width: thin; /* Firefox */
          scrollbar-color: #4a4e5a #2a2d37; /* Firefox */
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #2a2d37;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #4a4e5a;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #5a5e6a;
        }
        .custom-scrollbar {
          overflow-y: auto !important;
        }
        
        /* Fix text overflow */
        .break-words {
          word-wrap: break-word;
          word-break: break-word;
        }
      `}</style>
    </div>
  );
}

export default ChatArea;
