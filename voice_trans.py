import os
import gtts
from playsound import playsound
from googletrans import Translator

SUPPORTED_LANGUAGES = gtts.lang.tts_langs()

def text_to_speech(text, lang, output_file='output.mp3'):
    try:
        tts = gtts.gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        
        playsound(output_file)
    except Exception as e:
        print(f"Unexpected error with '{lang}': {e}")
        print("Please try another language.")

def translate_text(translator, text, lang):
    try:
        result = translator.translate(text, dest=lang)
        return result.text
    except Exception as e:
        print(f"Translation failed for '{lang}': {e}")
        return text

if __name__ == "__main__":
    translator = Translator()
    
    location = input("Enter the location for the news: ")

    text = input("Enter the news text: ").strip()
    
    if not text:
        print("No text provided.")
    else:
        intro = f"Welcome to the latest news update from {location}."
        outro = "Thank you for listening to today's news update. Stay informed and stay safe."
        
        full_text = f"{intro} {text} {outro}"
        
        while True:
            lang = input("Enter the language code for audio output (e.g., 'en' for English, 'fr' for French, 'es' for Spanish) or type 'exit' to cancel: ").strip().lower()

            if lang == "exit":
                print("Text-to-Speech process canceled.")
                break
            
            if lang not in SUPPORTED_LANGUAGES.keys():
                print(f"Error: '{lang}' is not a supported language for text-to-speech. Please try again.")
                continue
            
            translated_text = translate_text(translator, full_text, lang)
            
            print(f"\nTranscript in '{lang}': {translated_text}\n")
            
            text_to_speech(translated_text, lang)
            break