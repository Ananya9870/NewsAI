from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import os
import json
import time
import io
import requests
from typing import Optional, List, Dict, Any
import gtts

# Import NewsAgent class
from main import NewsAgent

app = FastAPI(
    title="NewsAI API",
    description="A FastAPI backend for a location-specific news agent that provides news based on pincode and preferred language.",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the NewsAgent
news_agent = NewsAgent()

# Session storage for user conversations
user_sessions = {}

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian"
}

# Request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    pincode: Optional[str] = None
    language: str = "en"

class PincodeRequest(BaseModel):
    pincode: str

class TextToSpeechRequest(BaseModel):
    text: str
    lang: str = "en"

class TranslateRequest(BaseModel):
    text: str
    target_language: str

class NewsResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    translated: bool = False

# Helper function to get or create a session
def get_session(session_id: str) -> Dict:
    """Get or create a user session by ID."""
    if session_id not in user_sessions:
        user_sessions[session_id] = {"location": None, "language": "en", "history": []}
    return user_sessions[session_id]

# Helper function to get location from pincode
def get_location_from_pincode(pincode: str) -> Optional[str]:
    """Get location (city/state) from Indian pincode."""
    try:
        # First try India Post API
        url = f"https://api.postalpincode.in/pincode/{pincode}"
        response = requests.get(url, timeout=5,verify=False)
        data = response.json()
        
        if data and data[0]["Status"] == "Success":
            post_office = data[0]["PostOffice"][0]
            district = post_office["District"]
            state = post_office["State"]
            return f"{district}, {state}"
        
        # Fallback to pincode map
        pincode_map = {
            "11": "Delhi",
            "12": "Haryana",
            "13": "Haryana",
            "14": "Punjab",
            "15": "Punjab",
            "16": "Punjab",
            "17": "Himachal Pradesh",
            "18": "Jammu & Kashmir",
            "19": "Jammu & Kashmir",
            "20": "Uttar Pradesh",
            "21": "Uttar Pradesh",
            "22": "Uttar Pradesh",
            "23": "Uttar Pradesh",
            "24": "Uttar Pradesh",
            "25": "Uttar Pradesh",
            "26": "Uttar Pradesh",
            "27": "Uttar Pradesh",
            "28": "Uttar Pradesh",
            "30": "Rajasthan",
            "31": "Rajasthan",
            "32": "Rajasthan",
            "33": "Rajasthan",
            "34": "Rajasthan",
            "36": "Gujarat",
            "37": "Gujarat",
            "38": "Gujarat",
            "39": "Gujarat",
            "40": "Maharashtra",
            "41": "Maharashtra",
            "42": "Maharashtra",
            "43": "Maharashtra",
            "44": "Maharashtra",
            "45": "Madhya Pradesh",
            "46": "Madhya Pradesh",
            "47": "Madhya Pradesh",
            "48": "Madhya Pradesh",
            "49": "Chhattisgarh",
            "50": "Andhra Pradesh",
            "51": "Andhra Pradesh",
            "52": "Telangana",
            "53": "Telangana",
            "56": "Karnataka",
            "57": "Karnataka",
            "58": "Karnataka",
            "59": "Karnataka",
            "60": "Tamil Nadu",
            "61": "Tamil Nadu",
            "62": "Tamil Nadu",
            "63": "Tamil Nadu",
            "64": "Tamil Nadu",
            "67": "Kerala",
            "68": "Kerala",
            "69": "Kerala",
            "70": "West Bengal",
            "71": "West Bengal",
            "72": "West Bengal",
            "73": "West Bengal",
            "74": "West Bengal",
            "75": "Odisha",
            "76": "Odisha",
            "77": "Odisha",
            "78": "Assam",
            "79": "North East India",
            "80": "Bihar",
            "81": "Bihar",
            "82": "Bihar",
            "83": "Jharkhand",
            "84": "Jharkhand",
            "85": "Jharkhand"
        }
        
        # Get state from first 2 digits
        state = pincode_map.get(pincode[:2], "Unknown")
        return state
        
    except Exception as e:
        print(f"Error getting location from pincode: {e}")
        return None

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "NewsAI API is running"}

@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages"""
    return {"languages": SUPPORTED_LANGUAGES}

@app.post("/api/pincode")
async def lookup_pincode(request: PincodeRequest):
    """Look up location from pincode"""
    location = get_location_from_pincode(request.pincode)
    if not location:
        raise HTTPException(status_code=404, detail="Could not find location for this pincode")
    return {"pincode": request.pincode, "location": location}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process a chat message and return a response in the requested language"""
    session = get_session(request.session_id)
    
    # Update language preference
    if request.language:
        session["language"] = request.language
    
    # Update location if pincode provided
    if request.pincode:
        location = get_location_from_pincode(request.pincode)
        if location:
            session["location"] = location
    
    # Process the query
    query = request.message
    
    # If location is set, include it in the query for location-specific news
    if session["location"] and "news" in query.lower() and session["location"].lower() not in query.lower():
        query = f"{query} in {session['location']}"
    
    # Process the query
    response = news_agent.process_query(query)
    
    # Translate response if needed
    original_response = response
    if session["language"] != "en":
        try:
            # Use the translate_text method from NewsAgent
            translation_input = json.dumps({"text": response, "lang": session["language"]})
            translated_response = news_agent.translate_text(translation_input)
            
            # Extract translated text from response format "Translated text: {text}"
            if "Translated text: " in translated_response:
                response = translated_response.replace("Translated text: ", "")
            else:
                response = translated_response
        except Exception as e:
            print(f"Translation error: {e}")
            # Keep original response if translation fails
    
    # Store in session history
    session["history"].append({"role": "user", "content": request.message})
    session["history"].append({"role": "assistant", "content": response})
    
    return {
        "response": response,
        "original_response": original_response if session["language"] != "en" else None,
        "language": session["language"],
        "location": session["location"]
    }

@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """Translate text to the specified language"""
    try:
        translation_input = json.dumps({"text": request.text, "lang": request.target_language})
        translated_text = news_agent.translate_text(translation_input)
        
        # Extract translated text from response format "Translated text: {text}"
        if "Translated text: " in translated_text:
            translated_text = translated_text.replace("Translated text: ", "")
        
        return {"translated_text": translated_text, "language": request.target_language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech and return audio file"""
    try:
        # Generate speech
        tts = gtts.gTTS(text=request.text, lang=request.lang, slow=False)
        
        # Save to in-memory file
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        # Return audio file
        return StreamingResponse(
            audio_io,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/api/news/{location}")
async def get_location_news(location: str, count: int = 5, language: str = "en"):
    """Fetch news for a specific location and optionally translate it"""
    try:
        # Fetch news
        news = news_agent.fetch_city_news(f"{location}, {count}")
        
        # Translate if needed
        if language != "en":
            translation_input = json.dumps({"text": news, "lang": language})
            translated_news = news_agent.translate_text(translation_input)
            
            # Extract translated text
            if "Translated text: " in translated_news:
                news = translated_news.replace("Translated text: ", "")
            else:
                news = translated_news
        
        return {"news": news, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")
