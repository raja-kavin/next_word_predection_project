
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os

# Small sample dataset
sentences = [
    "i love python programming",
    "i love machine learning",
    "i love deep learning",
    "i love data science",
    "python is a programming language",
    "python is easy to learn",
    "machine learning is interesting",
    "deep learning is powerful",
    "data science is interesting",
    "artificial intelligence is powerful",
    "i am learning python",
    "i am learning machine learning",
    "i am learning deep learning",
    "python is used for data science",
    "machine learning uses data",
    "deep learning uses neural networks",
    "i want to learn python",
    "i want to learn data science",
    "i want to learn artificial intelligence",
    "neural networks are used in deep learning"
]

# Repeat sentences to create more training examples.
# Note: Repetition does not add new information.
sentences = sentences * 20

# Convert words into numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

vocab_size = len(tokenizer.word_index) + 1

# Create input sequences
sequences = []

for sentence in sentences:
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    print('Actual token')
    print(tokenizer.texts_to_sequences([sentence])[0])

    for i in range(1, len(tokens)):
        print(f"TOKEN {i}", tokens[:i + 1])
        sequences.append(tokens[:i + 1])

print('Whole sequence:')
print(sequences)

# Make all sequences the same length
max_length = max(len(seq) for seq in sequences)

sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="pre"
)

# X = input words
# y = next word
'''

word = [i, love, python]

padding sequence output = [[ 0  0  0  2  7  4]]

sequences[:, :-1] -> gives ->  [ 0  0  0  2  7] 
sequences[:, -1] -> gives [4]
'''
X = sequences[:, :-1]
y = sequences[:, -1]

print('Input:')
print(X)

print('Output:')
print(y)

# Split dataset (same split for both models)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Simple RNN model
rnn_model = Sequential([
    Embedding(vocab_size, 16),
    SimpleRNN(32),
    Dense(vocab_size, activation="softmax")
])

# LSTM model
lstm_model = Sequential([
    Embedding(vocab_size, 16),
    LSTM(32),
    Dense(vocab_size, activation="softmax")
])

# Compile both models
for model in [rnn_model, lstm_model]:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

# Train RNN
print("Training RNN...")
rnn_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    verbose=1
)

# Train LSTM
print("Training LSTM...")
lstm_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    verbose=1
)

# Evaluate both models
rnn_train_loss, rnn_train_acc = rnn_model.evaluate(
    X_train, y_train, verbose=0
)

lstm_train_loss, lstm_train_acc = lstm_model.evaluate(
    X_train, y_train, verbose=0
)

rnn_loss, rnn_acc = rnn_model.evaluate(
    X_test, y_test, verbose=0
)

lstm_loss, lstm_acc = lstm_model.evaluate(
    X_test, y_test, verbose=0
)

print("RNN Training Accuracy:", round(rnn_train_acc * 100, 2))
print("LSTM Training Accuracy:", round(lstm_train_acc * 100, 2))
print("RNN Test Accuracy:", round(rnn_acc * 100, 2))
print("LSTM Test Accuracy:", round(lstm_acc * 100, 2))

# Save models
os.makedirs("models", exist_ok=True)

rnn_model.save("models/rnn.keras")
lstm_model.save("models/lstm.keras")

# Save tokenizer and accuracy
# .pkl Save Python objects using Pickle
with open("models/data.pkl", "wb") as file:
    pickle.dump({
        "tokenizer": tokenizer,
        "max_length": max_length,
        "rnn_accuracy": rnn_acc * 100,
        "lstm_accuracy": lstm_acc * 100,
        "rnn_train_accuracy": rnn_train_acc * 100,
        "lstm_train_accuracy": lstm_train_acc * 100
    }, file)

print("Models saved successfully!")

# ============================================================
# 1. OPENAI GPT API
# ============================================================
# Create an OpenAI API key:
# 1. Visit https://platform.openai.com/
# 2. Sign in to your OpenAI account.
# 3. Visit https://platform.openai.com/api-keys
# 4. Select a project and click "Create new secret key".
# 5. Copy the key and add it to your .env file.
# 6. Check API billing/credits: https://platform.openai.com/settings/organization/billing/overview
#
# Install:
# pip install openai python-dotenv
#
# .env:
# OPENAI_API_KEY=your_openai_api_key

# from dotenv import load_dotenv
# from openai import OpenAI
# import os

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")

# if not api_key:
#     raise ValueError("OPENAI_API_KEY is missing from the .env file.")

# try:
#     client = OpenAI(api_key=api_key)

#     # List models visible to this API account.
#     models = client.models.list()
#     model_names = [model.id for model in models.data]

#     print("Available OpenAI models:")
#     for name in model_names:
#         print(name)

#     # These are candidates, not guaranteed-free models.
#     preferred_models = ["gpt-5-mini", "gpt-4.1-mini"]

#     for model_name in preferred_models:
#         if model_name not in model_names:
#             continue

#         try:
#             response = client.responses.create(
#                 model=model_name,
#                 input="What is Machine Learning? Explain briefly."
#             )
#             print(f"\nModel used: {model_name}")
#             print(response.output_text)
#             break
#         except Exception as error:
#             print(f"{model_name} failed: {error}")
#     else:
#         print("None of the preferred models generated a response.")

# except Exception as error:
#     print(f"OpenAI API error: {error}")


# ============================================================
# 2. GOOGLE GEMINI API
# ============================================================
# Create a Gemini API key:
# 1. Visit https://aistudio.google.com/
# 2. Sign in with your Google account.
# 3. Visit https://aistudio.google.com/api-keys
# 4. Create/select a Google Cloud project if prompted.
# 5. Create the key, copy it, and add it to your .env file.
# 6. Check available models and usage limits in AI Studio.
#
# Install:
# pip install google-genai python-dotenv
#
# .env:
# GEMINI_API_KEY=your_gemini_api_key

# from dotenv import load_dotenv
# from google import genai
# import os

# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY is missing from the .env file.")

# try:
#     client = genai.Client(api_key=api_key)

#     # List models visible to this API account.
#     models = client.models.list()
#     model_names = []

#     print("Available Gemini models:")
#     for model in models:
#         name = model.name.replace("models/", "")
#         print(name)
#         model_names.append(name)

#     # These are potential candidates; free-tier access depends on account/quota.
#     preferred_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

#     for model_name in preferred_models:
#         if model_name not in model_names:
#             continue

#         try:
#             response = client.models.generate_content(
#                 model=model_name,
#                 contents="What is Deep Learning? Explain briefly."
#             )
#             if response.text:
#                 print(f"\nModel used: {model_name}")
#                 print(response.text)
#                 break
#             print(f"{model_name} returned no text.")
#         except Exception as error:
#             print(f"{model_name} failed: {error}")
#     else:
#         print("None of the preferred models generated a response.")

# except Exception as error:
#     print(f"Gemini API error: {error}")


# # ============================================================
# # 3. META LLAMA API
# # ============================================================
# # Important: Meta's Llama model/API access and pricing depend on the
# # service/provider you use. Do not assume that a Llama API is free.
# #
# # Start here:
# # Meta AI: https://www.meta.ai/
# # Llama developer information: https://llama.developer.meta.com/
# # The Llama API client package may be provider-specific. Follow the
# # official instructions for the account/provider that issued your key.
# #
# # Install:
# # pip install llama-api-client python-dotenv
# #
# # .env:
# # LLAMA_API_KEY=your_llama_api_key

# from dotenv import load_dotenv
# from llama_api_client import LlamaAPIClient
# import os

# load_dotenv()

# api_key = os.getenv("LLAMA_API_KEY")

# if not api_key:
#     raise ValueError("LLAMA_API_KEY is missing from the .env file.")

# try:
#     client = LlamaAPIClient(api_key=api_key)

#     # This method is available only in SDK versions/providers that support it.
#     models_response = client.models.list()
#     model_names = [model.id for model in models_response.data]

#     print("Available Llama models:")
#     for name in model_names:
#         print(name)

#     # Try listed models in sequence. Availability does not mean free access.
#     for model_name in model_names:
#         try:
#             response = client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": "What is Artificial Intelligence? Explain briefly."
#                     }
#                 ]
#             )
#             print(f"\nModel used: {model_name}")
#             print(response.completion_message.content)
#             break
#         except Exception as error:
#             print(f"{model_name} failed: {error}")
#     else:
#         print("No listed model generated a response.")

# except Exception as error:
#     print(f"Llama API error: {error}")


# # ============================================================
# # 4. PROMPT ENGINEERING WITH GEMINI
# # ============================================================
# # Create a Gemini API key:
# # 1. Visit https://aistudio.google.com/
# # 2. Sign in.
# # 3. Visit https://aistudio.google.com/api-keys
# # 4. Create a key and add it to .env as GEMINI_API_KEY.
# #
# # Install:
# # pip install google-genai python-dotenv

# from dotenv import load_dotenv
# from google import genai
# import os

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY is missing from the .env file.")

# prompt = """
# Act as an AI trainer.
# Explain supervised learning in simple English.
# Give one definition and one real-world example.
# """

# try:
#     client = genai.Client(api_key=api_key)
#     models = client.models.list()
#     model_names = [model.name.replace("models/", "") for model in models]

#     print("Available Gemini models:")
#     for name in model_names:
#         print(name)

#     preferred_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

#     for model_name in preferred_models:
#         if model_name not in model_names:
#             continue

#         try:
#             response = client.models.generate_content(
#                 model=model_name,
#                 contents=prompt
#             )
#             if response.text:
#                 print(f"\nModel used: {model_name}")
#                 print(response.text)
#                 break
#         except Exception as error:
#             print(f"{model_name} failed: {error}")
#     else:
#         print("No preferred model generated a response.")

# except Exception as error:
#     print(f"Gemini API error: {error}")


# ============================================================
# 5. SYSTEM (MACHINE) PROMPT AND USER PROMPT WITH GEMINI
# ============================================================
# Create a Gemini API key:
# 1. Visit https://aistudio.google.com/
# 2. Sign in.
# 3. Visit https://aistudio.google.com/api-keys
# 4. Create a key and add it to .env as GEMINI_API_KEY.
#
# Install:
# pip install google-genai python-dotenv

# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# import os

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY is missing from the .env file.")

# system_prompt = """
# You are a Data Science trainer.
# Explain concepts in simple English.
# Give one real-world example.
# Keep answers short.
# """

# user_prompt = "Explain classification in Machine Learning."

# try:
#     client = genai.Client(api_key=api_key)
#     models = client.models.list()
#     model_names = [model.name.replace("models/", "") for model in models]

#     print("Available Gemini models:")
#     for name in model_names:
#         print(name)

#     preferred_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

#     for model_name in preferred_models:
#         if model_name not in model_names:
#             continue

#         try:
#             response = client.models.generate_content(
#                 model=model_name,
#                 contents=user_prompt,
#                 config=types.GenerateContentConfig(
#                     system_instruction=system_prompt
#                 )
#             )
#             if response.text:
#                 print(f"\nModel used: {model_name}")
#                 print(response.text)
#                 break
#         except Exception as error:
#             print(f"{model_name} failed: {error}")
#     else:
#         print("No preferred model generated a response.")

# except Exception as error:
#     print(f"Gemini API error: {error}")


# # ============================================================
# # 6. MINI PROJECT: AI DATA SCIENCE TUTOR WITH GEMINI
# # ============================================================
# # Create a Gemini API key:
# # 1. Visit https://aistudio.google.com/
# # 2. Sign in.
# # 3. Visit https://aistudio.google.com/api-keys
# # 4. Create a key and add it to .env as GEMINI_API_KEY.
# #
# # Install:
# # pip install google-genai python-dotenv

# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# import os

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY is missing from the .env file.")

# system_prompt = """
# You are a Data Science tutor.
# 1. Explain concepts in simple English.
# 2. Give one real-world example.
# 3. Focus on Data Science-related questions.
# 4. Keep answers short.
# """

# user_prompt = input("Ask a Data Science question: ")

# try:
#     client = genai.Client(api_key=api_key)
#     models = client.models.list()
#     model_names = [model.name.replace("models/", "") for model in models]

#     print("\nAvailable Gemini models:")
#     for name in model_names:
#         print(name)

#     preferred_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

#     for model_name in preferred_models:
#         if model_name not in model_names:
#             continue

#         try:
#             response = client.models.generate_content(
#                 model=model_name,
#                 contents=user_prompt,
#                 config=types.GenerateContentConfig(
#                     system_instruction=system_prompt
#                 )
#             )
#             if response.text:
#                 print(f"\nModel used: {model_name}")
#                 print("\nAI Tutor:", response.text)
#                 break
#         except Exception as error:
#             print(f"{model_name} failed: {error}")
#     else:
#         print("No preferred model generated a response.")

# except Exception as error:
#     print(f"Gemini API error: {error}")
