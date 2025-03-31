import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os
import base64

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

def autoplay_audio(file_path: str):
    """Autoplay the audio file in Streamlit."""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

def speak_response(text):
    """Generate speech from text using gTTS and play in Streamlit."""
    try:
        tts = gTTS(text=text, lang="en")
        audio_file = "response.mp3"
        tts.save(audio_file)
        
        # Auto-play the audio
        autoplay_audio(audio_file)
        
        # Clean up the audio file after use
        if os.path.exists(audio_file):
            os.remove(audio_file)

    except Exception as e:
        st.error(f"Error in generating speech: {str(e)}")

# Streamlit UI
st.title("🤖 Emotion-Aware Chatbot")
st.write("Type a message, and I'll respond based on the emotion I detect!")

# Text input
user_input = st.text_input("Type your message here:", key="text_input")

# Process input and generate response
if user_input.strip():
    st.write(f"**You:** {user_input}")
    
    bot_response = detect_emotion(user_input)
    st.write(f"**Chatbot:** {bot_response}")
    
    # Generate and play speech response
    if st.checkbox("Enable voice response", value=True):
        speak_response(bot_response)
