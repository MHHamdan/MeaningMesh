
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
	padding_type = "<OOV>"

	# YOUR CODE HERE
	tokenizer = Tokenizer(num_words=vocab_size, oov_token=padding_type)
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)
	sequences = pad_sequences(sequences, maxlen=max_length, padding=trunc_type)

	X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	model = tf.keras.models.Sequential([
		# YOUR CODE HERE
		tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
		tf.keras.layers.Dense(1, activation="sigmoid")
	])

	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
	model.fit(X_train, y_train, epochs=6, batch_size=128, validation_data=(X_test, y_test))
	return model


# You must save your model in the .h5 format like this before submitting.
# Google expect that to test your solution.
if __name__ == "__main__":
	model = solution_model()
	model.save("mymodel.h5")
