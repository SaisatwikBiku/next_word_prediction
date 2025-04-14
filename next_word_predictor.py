import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# 1. Data Loading and Preprocessing (Using IMDB Reviews)
# =============================================================================

st.write("### Loading and Preprocessing IMDB Reviews Dataset")
st.write("This may take a couple of minutes...")

# Load the IMDB Reviews dataset (train split)
ds_train = tfds.load("imdb_reviews", split="train", shuffle_files=True)

# Concatenate the 'text' field from a limited number of examples.
text_list = []
max_examples = 2000  # Adjust as needed: more examples may improve accuracy but require more time.
for i, example in enumerate(tfds.as_numpy(ds_train)):
    text = example['text'].decode('utf-8')
    text_list.append(text)
    if i >= max_examples - 1:
        break

full_text = "\n".join(text_list)
st.write("Dataset loaded. Total characters:", len(full_text))

# Tokenize the text on the word level.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([full_text])
total_words = len(tokenizer.word_index) + 1
st.write("Vocabulary size:", total_words)

# Convert full_text to a sequence of token indices.
tokens = tokenizer.texts_to_sequences([full_text])[0]
st.write("Total tokens before limiting:", len(tokens))

# Limit the tokens for faster training.
max_tokens = 20000  
tokens = tokens[:max_tokens]
st.write("Using {} tokens for training.".format(len(tokens)))

# Set a fixed sequence length (input + target).
max_sequence_len = 20  # You can adjust this value

# Generate training sequences using a sliding window.
input_sequences = []
for i in range(max_sequence_len, len(tokens) + 1):
    seq = tokens[i - max_sequence_len:i]
    input_sequences.append(seq)
input_sequences = np.array(input_sequences)
st.write("Total training sequences:", input_sequences.shape[0])

# Split each sequence into predictors (X) and label (y).
X = input_sequences[:, :-1]  # Input sequence (all tokens except the last)
y = input_sequences[:, -1]   # Target word (last token)

# One-hot encode the labels.
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# =============================================================================
# 2. Build the LSTM-based Next-Word Prediction Model
# =============================================================================

st.write("### Building LSTM-based Next Word Prediction Model")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=total_words, output_dim=100, input_length=(max_sequence_len - 1)),
    tf.keras.layers.LSTM(150, return_sequences=False),
    tf.keras.layers.Dense(total_words, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary(print_fn=st.text)

# =============================================================================
# 3. Train the Model (Overnight) or Load Existing Checkpoint
# =============================================================================

st.write("### Training the Model / Loading Saved Weights")
epochs = 100  
batch_size = 64
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "lstm_best_model.h5")

# Set up a checkpoint callback to save the best model.
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="accuracy",
    save_best_only=True,
    verbose=1,
)

if os.path.exists(checkpoint_path):
    st.write("Loading saved model weights from checkpoint.")
    model.load_weights(checkpoint_path)
    st.write("Model weights loaded. Skipping training.")
else:
    st.write("No saved model found. Training from scratch...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    st.write("Training complete.")

# =============================================================================
# 4. Next-Word Prediction Function
# =============================================================================

def predict_next_words(seed_text, next_words=1):
    """
    Given a seed text, predict the next `next_words` tokens iteratively using the trained LSTM model.
    """
    for _ in range(next_words):
        # Convert the seed text to a sequence of tokens.
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequence to the fixed length (input length expected by the model).
        token_list = pad_sequences([token_list], maxlen=(max_sequence_len - 1), padding='pre')
        # Predict the next word.
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=1)[0]
        # If predicted_index is 0 (unknown), break.
        if predicted_index == 0:
            break
        # Map the predicted index back to a word.
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                seed_text += " " + word
                break
    return seed_text

# =============================================================================
# 5. Streamlit Web Interface for Prediction
# =============================================================================

st.title("Next Word Predictor (Embedding + LSTM)")
st.write("After training, enter your seed text and specify the number of words to predict:")

seed_text_input = st.text_input("Seed text", "The movie was")
num_words_input = st.number_input("Words to predict", min_value=1, max_value=20, value=1, step=1)

if st.button("Predict"):
    st.write("Generating predictions...")
    prediction = predict_next_words(seed_text_input, int(num_words_input))
    st.write("### Prediction Output:")
    st.write(prediction)
