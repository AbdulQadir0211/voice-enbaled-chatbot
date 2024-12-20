from flask import Flask, request, jsonify,render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
import speech_recognition as sr
import torch

app = Flask(__name__)

# Load Chatbot Model
model_name = "model\microsoftDialoGPT"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

chat_history = []

def chat_with_bot(user_input, chat_history=[]):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([torch.tensor(chat_history), input_ids], dim=-1) if chat_history else input_ids
    
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/stt', methods=['POST'])
def speech_to_text():
    recognizer = sr.Recognizer()
    try:
        # Save uploaded audio file
        audio_file = request.files['audio']
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)

        # Convert speech to text
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.json
    user_input = data.get("text", "")
    
    if not user_input:
        return jsonify({"error": "No text provided"}), 400
    
    response, chat_history = chat_with_bot(user_input, chat_history)
    return jsonify({"response": response})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    response_text = data.get("text", "")
    
    if not response_text:
        return jsonify({"error": "No text provided"}), 400
    
    tts = gTTS(text=response_text, lang='en')
    audio_path = "response_audio.mp3"
    tts.save(audio_path)
    return jsonify({"audio_url": f"/audio/{audio_path}"})

@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    return app.send_static_file(filename)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
