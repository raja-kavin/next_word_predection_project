
from flask import Flask, render_template, request
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained models
rnn_model = load_model("models/rnn.keras")
lstm_model = load_model("models/lstm.keras")

# Load tokenizer and accuracy
with open("models/data.pkl", "rb") as file:
    data = pickle.load(file)

tokenizer = data["tokenizer"]
max_length = data["max_length"]

rnn_accuracy = round(data["rnn_accuracy"], 2)
lstm_accuracy = round(data["lstm_accuracy"], 2)

# Predict the next word
def predict_next_word(model, sentence):

    # Convert input sentence into numbers
    sequence = tokenizer.texts_to_sequences([sentence])[0]

    # Match model input length
    sequence = pad_sequences(
        [sequence],
        maxlen=max_length - 1,
        padding="pre",
        truncating="pre"
    )

    # Predict probabilities for all words
    prediction = model.predict(sequence, verbose=0)

    # Get word with highest probability
    word_id = np.argmax(prediction[0])

    # Convert number back into word
    next_word = tokenizer.index_word.get(word_id, "Unknown")

    return next_word


@app.route("/", methods=["GET", "POST"])
def home():

    rnn_word = None
    lstm_word = None
    sentence = ""

    if request.method == "POST":

        sentence = request.form["sentence"]

        if sentence.strip():
            rnn_word = predict_next_word(rnn_model, sentence)
            lstm_word = predict_next_word(lstm_model, sentence)

    return render_template(
        "index.html",
        sentence=sentence,
        rnn_word=rnn_word,
        lstm_word=lstm_word,
        rnn_accuracy=rnn_accuracy,
        lstm_accuracy=lstm_accuracy
    )


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pickle

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = Flask(__name__)

# # Load saved models
# rnn_model = load_model("models/rnn.keras")
# lstm_model = load_model("models/lstm.keras")

# # Load tokenizer and other information
# with open("models/data.pkl", "rb") as file:
#     data = pickle.load(file)

# tokenizer = data["tokenizer"]
# max_length = data["max_length"]


# def predict_next_word(model, sentence):

#     # Convert the typed sentence into word IDs
#     sequence = tokenizer.texts_to_sequences([sentence])[0]

#     # Unknown words cannot be understood by this tokenizer
#     if len(sequence) == 0:
#         return ""

#     # Keep the same input length used during training
#     sequence = pad_sequences(
#         [sequence],
#         maxlen=max_length - 1,
#         padding="pre",
#         truncating="pre"
#     )

#     # Predict the next word
#     probabilities = model.predict(sequence, verbose=0)[0]

#     word_id = int(np.argmax(probabilities))

#     # Convert the predicted ID into a word
#     return tokenizer.index_word.get(word_id, "")


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():

#     sentence = request.json.get("sentence", "").strip()

#     if not sentence:
#         return jsonify({
#             "rnn": "",
#             "lstm": ""
#         })

#     # Get predictions from both models
#     rnn_word = predict_next_word(rnn_model, sentence)
#     lstm_word = predict_next_word(lstm_model, sentence)

#     return jsonify({
#         "rnn": rnn_word,
#         "lstm": lstm_word
#     })


# if __name__ == "__main__":
#     app.run(debug=True)
