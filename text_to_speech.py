from gtts import gTTS
import os

def speak_response(response_text):
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  # Use "mpg321 response.mp3" for Linux
