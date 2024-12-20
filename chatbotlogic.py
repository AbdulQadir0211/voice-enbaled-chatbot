from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Chatbot Model
model_name = "model\microsoftDialoGPT"  # Replace with your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize conversation history
chat_history = []

def chat_with_bot(user_input, chat_history=[]):
    """
    Generates a chatbot response based on the user's input.
    
    Args:
        user_input (str): The text input from the user.
        chat_history (list): The conversation history with the bot.

    Returns:
        response (str): The chatbot's response.
        chat_history (list): Updated conversation history.
    """
    # Tokenize user input and add an end-of-sequence token
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Append user input to chat history
    bot_input_ids = torch.cat([torch.tensor(chat_history), input_ids], dim=-1) if chat_history else input_ids

    # Generate a response
    chat_history = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Adjust for creativity (higher = more creative)
        top_k=50,         # Limit responses to top-k most likely words
        top_p=0.9         # Nucleus sampling for diverse responses
    )

    # Decode and return the response
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history
