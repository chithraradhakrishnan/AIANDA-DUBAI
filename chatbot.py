import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import os
import nest_asyncio

# Apply nest_asyncio to handle async conflicts
nest_asyncio.apply()

# Cache the emotion classifier model to prevent reloading
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Load the pre-trained emotion detection model
emotion_classifier = load_emotion_model()

# Emotion-based responses
responses = {
    "joy": "I'm glad to hear that! 😊",
    "sadness": "I'm here for you. Things will get better. 💙",
    "anger": "I understand. Take a deep breath. Let's talk about it. 😌",
    "enthusiasm": "I love your energy! Let's go! 🚀",
    "neutral": "That's interesting. I'd love to hear more! 🤖",
    "fear": "I'm here if you want to talk. Stay strong! 💪",
    "surprise": "Wow, that's surprising! Tell me more! 😲"
}

def detect_emotion(text):
    """Function to detect emotion from text."""
    try:
        result = emotion_classifier(text)
        if result and isinstance(result, list) and "label" in result[0]:  
            label = result[0]["label"].lower()
        else:
            label = "neutral"  

        return responses.get(label, "That's an interesting point! Could you elaborate? 🤖")

    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return "I'm having trouble understanding. Could you try again?"

def speak_response(text):
    """Generate speech from text using gTTS and play in Streamlit."""
    try:
        tts = gTTS(text=text, lang="en")
        audio_file = "response.mp3"
        tts.save(audio_file)  # Save audio file

        # Streamlit audio player
        st.audio(audio_file, format="audio/mp3")
        
        # Clean up the audio file after use
        if os.path.exists(audio_file):
            os.remove(audio_file)

    except Exception as e:
        st.error(f"Error in generating speech: {str(e)}")

def get_voice_input():
    """Function to get voice input and convert to text."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Please speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand."
            except sr.RequestError:
                return "Could not request results. Please check your internet connection."
    except Exception as e:
        st.error(f"Microphone error: {str(e)}")
        return "Microphone not available."

# Streamlit UI
st.title("🤖 Voice-Enabled Emotion-Aware Chatbot")
st.write("Type or speak a message, and I'll respond based on your emotions!")

# Initialize session state for user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Input methods
input_method = st.radio("Input method:", ("Type", "Speak"), horizontal=True)

if input_method == "Speak":
    if st.button("🎤 Start Recording"):
        voice_input = get_voice_input()
        if voice_input and voice_input not in ["Sorry, I couldn't understand.", 
                                             "Could not request results. Please check your internet connection.",
                                             "Microphone not available."]:
            st.session_state.user_input = voice_input
            st.rerun()  # Refresh to show the spoken input
        else:
            st.warning(voice_input)
else:
    user_input = st.text_input("Type your message here:", 
                             key="text_input", 
                             value=st.session_state.user_input)
    if user_input:
        st.session_state.user_input = user_input

# Process input and generate response
if st.session_state.user_input.strip():
    st.write(f"*You:* {st.session_state.user_input}")
    
    bot_response = detect_emotion(st.session_state.user_input)
    st.write(f"*Chatbot:* {bot_response}")
    
    # Generate and play speech response
    speak_response(bot_response)