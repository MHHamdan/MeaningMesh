
"""
NLP QUESTION

Build and train a classifier for Clickbait detection!

Create a binary classifier based on RNN architecture that takes a headline
and predicts if it's a clickbait.

https://github.com/bhargaviparanjape/clickbait
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split


def dataset_loader():
	sentences = []
	labels = []

	with open("data/clickbait_data", "r", encoding='latin-1') as f:
		for line in f.readlines():
			sentences.append(line)
			labels.append(1)

	with open("data/non_clickbait_data", "r", encoding='latin-1') as f:
		for line in f.readlines():
			sentences.append(line)
			labels.append(0)

	return sentences, labels


def solution_model():
    sentences, labels = dataset_loader()

    # DO NOT CHANGE THIS CODE OR TEST MAY NOT WORK
    vocab_size = 3000
    embedding_dim = 50
    max_length = 100
    trunc_type = "post"
    padding_type = "post"
    oov_token = "<OOV>"

    # Tokenization
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Define the RNN-based model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, np.array(y_train), epochs=10, validation_data=(X_test, np.array(y_test)), verbose=1)

    return model



# You must save your model in the .h5 format like this before submitting.
# Google expect that to test your solution.
if __name__ == "__main__":
	model = solution_model()
	model.save("mymodel101.h5")
