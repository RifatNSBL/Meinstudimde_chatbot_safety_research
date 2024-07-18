

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load list of inappropriate words from a file
def load_bad_words(filepath):
    with open(filepath, 'r') as file:
        bad_words = file.read().splitlines()
    return bad_words

# Path to the file with inappropriate words (update this path)
bad_words_filepath = 'path_to_bad_words_file.txt'
bad_words_list = load_bad_words(bad_words_filepath)

# Function to preprocess messages
def preprocess_message(message):
    inputs = tokenizer(message, return_tensors='pt', truncation=True, padding=True, max_length=512)
    return inputs

# Function to classify messages with BERT
def classify_message(message):
    inputs = preprocess_message(message)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Keyword-based filtering function
def filter_message(message, bad_words):
    message = message.lower()
    for word in bad_words:
        if re.search(r'\b' + re.escape(word) + r'\b', message):
            return False, f"Inappropriate content detected: {word}"
    return True, "Message is appropriate"

# Chatbot response generation considering keyword filtering and BERT
def chatbot_response(user_input, bad_words):
    # Keyword-based filtering
    is_valid, response = filter_message(user_input, bad_words)
    if not is_valid:
        return response
    
    # BERT-based classification
    predicted_class = classify_message(user_input)
    if predicted_class == 1:  # Assume class 1 represents inappropriate content
        return "Your message contains inappropriate content and has been blocked."
    
    # Forward message to Mistral model
    mistral_response = mistral_generate_response(user_input)
    return mistral_response

# Placeholder function for Mistral model response generation
def mistral_generate_response(user_input):
    # Here you would integrate your Mistral model logic
    return "Response from Mistral chatbot."

# Test chatbot workflow
user_input = "This is a test message."
response = chatbot_response(user_input, bad_words_list)
print(response)
