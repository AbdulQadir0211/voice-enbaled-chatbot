import speech_recognition as sr

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            query = recognizer.recognize_google(audio)
            print(f"User: {query}")
            return query
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return "Sorry, I couldn't understand that."
        except sr.RequestError as e:
            print(f"STT Error: {e}")
            return "Sorry, I couldn't process your audio."



if __name__ == "__main__":
    while True:
        user_input = get_voice_input()  # Get voice input from the user
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break