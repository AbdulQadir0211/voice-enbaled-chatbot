from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from speech_to_text import get_voice_input
from text_to_speech import speak_response
from chatbotlogic import chat_with_bot

# Load Chatbot Model
model_name = "model\microsoftDialoGPT"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

chat_history = []

'''def chat_with_bot(user_input, chat_history=[]):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([torch.tensor(chat_history), input_ids], dim=-1) if chat_history else input_ids
    
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history'''

# Main Function
if __name__ == "__main__":
    while True:
        user_input = get_voice_input()  # Get voice input from the user
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        
        response, chat_history = chat_with_bot(user_input, chat_history)  # Generate chatbot response
        print(f"Bot: {response}")
        speak_response(response)  # Speak the response
