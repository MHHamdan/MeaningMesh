import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def dataset_loader():
    sentences = []
    labels = []

    with open("data/clickbait_data", "r", encoding='latin-1') as f:
        for line in f.readlines():
            sentences.append(line.strip())
            labels.append(1)

    with open("data/non_clickbait_data", "r", encoding='latin-1') as f:
        for line in f.readlines():
            sentences.append(line.strip())
            labels.append(0)

    return sentences, labels

def solution_model():
    sentences, labels = dataset_loader()

    # Tokenization and padding
    vocab_size = 3000
    embedding_dim = 50
    max_length = 100
    trunc_type = "post"
    padding_type = "post"
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(padded, labels, test_size=0.2, random_state=42)

    # Define the RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, np.array(train_labels), epochs=5, batch_size=128,  validation_data=(test_data, np.array(test_labels)))

    return model

if __name__ == "__main__":
    model = solution_model()
    model.save("model.h5")
