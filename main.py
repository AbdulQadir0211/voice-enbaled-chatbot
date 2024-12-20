import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
import speech_recognition as sr
import torch
import tempfile

# Load the pre-trained model
model_name = "model\microsoftDialoGPT"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize chat history
chat_history = []

# Function to process chat with the model
'''def chat_with_bot(user_input, chat_history=[]):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([torch.tensor(chat_history), input_ids], dim=-1) if chat_history else input_ids
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history'''

def chat_with_bot(user_input, chat_history=[]):
    # Encode user input and add EOS token
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Convert chat_history to tensor if it's not empty, otherwise use only the input_ids
    if chat_history:
        bot_input_ids = torch.cat([torch.tensor(chat_history), input_ids], dim=-1)  # Concatenate input with chat history
    else:
        bot_input_ids = input_ids  # Just use the input ids if history is empty
    
    # Generate response from the model
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the response
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response, chat_history


# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")  # Display feedback while listening
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Timeout after 5 seconds if no speech
    text = ""
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.write("Sorry, I did not understand that.")
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
    return text

# Function to convert text to speech
def text_to_speech(response_text):
    tts = gTTS(text=response_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

# Streamlit Interface
st.title("Voice Enabled Chatbot")
st.subheader("Talk to the chatbot and get responses in speech")

# Add stop button
stop_conversation = st.button("Stop Conversation")

# Real-time conversation loop
if st.button("Start Conversation") and not stop_conversation:
    chat_history = []  # Reset chat history
    while True:
        if stop_conversation:
            st.write("Conversation stopped.")
            break

        try:
            # Capture user input via microphone
            user_input = speech_to_text()
            if user_input.lower() == "stop":
                st.write("Stop command received. Ending conversation.")
                break

            if user_input:  # Only proceed if the speech is recognized
                st.write(f"You said: {user_input}")

                # Get chatbot response
                response, chat_history = chat_with_bot(user_input, chat_history)
                st.write(f"Bot says: {response}")

                # Convert text to speech and play the response automatically
                audio_path = text_to_speech(response)
                audio_file = open(audio_path, 'rb')
                st.audio(audio_file, format="audio/mp3")
            else:
                st.write("I couldn't hear anything. Please speak clearly.")

        except Exception as e:
            st.write(f"Error: {e}")
