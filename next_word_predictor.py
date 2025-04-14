import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# Sidebar: App Info and Instructions
# =============================================================================
st.sidebar.title("Next Word Predictor")
st.sidebar.markdown("""
**Overview:**
- This app trains an LSTM model for next-word prediction using a subset of IMDB Reviews.
- If a saved checkpoint exists, the model weights will load automatically.
- You can leave the training running overnight to improve accuracy.

**Usage:**
1. Train the model (or load existing weights).
2. Enter seed text and choose how many words to predict.
3. View prediction output.
""")
st.sidebar.info("Make sure to leave the app running to let the model train if no checkpoint is found.")

# =============================================================================
# Main Title and Description
# =============================================================================
st.title("Next Word Predictor (Embedding + LSTM)")
st.markdown("""
This application uses an LSTM network with an embedding layer for next-word prediction.
If you have already trained the model and a checkpoint exists, the model will load from the checkpoint.
Otherwise, it will train from scratch for the configured number of epochs.
""")

# =============================================================================
# 1. Data Loading and Preprocessing (Using IMDB Reviews)
# =============================================================================
st.header("Data Loading and Preprocessing")
st.write("Loading and preprocessing IMDB Reviews dataset. This may take a couple of minutes...")

ds_train = tfds.load("imdb_reviews", split="train", shuffle_files=True)
text_list = []
max_examples = 2000  # Adjust as needed: more examples may improve accuracy but require more time.
for i, example in enumerate(tfds.as_numpy(ds_train)):
    text = example['text'].decode('utf-8')
    text_list.append(text)
    if i >= max_examples - 1:
        break

full_text = "\n".join(text_list)
st.write(f"Dataset loaded. Total characters: {len(full_text)}")

tokenizer = Tokenizer()
tokenizer.fit_on_texts([full_text])
total_words = len(tokenizer.word_index) + 1
st.write(f"Vocabulary size: {total_words}")

tokens = tokenizer.texts_to_sequences([full_text])[0]
st.write(f"Total tokens before limiting: {len(tokens)}")

max_tokens = 20000  
tokens = tokens[:max_tokens]
st.write(f"Using {len(tokens)} tokens for training.")

max_sequence_len = 20  # Adjust as desired.
input_sequences = []
for i in range(max_sequence_len, len(tokens) + 1):
    seq = tokens[i - max_sequence_len:i]
    input_sequences.append(seq)
input_sequences = np.array(input_sequences)
st.write(f"Total training sequences: {input_sequences.shape[0]}")

X = input_sequences[:, :-1]  # Input sequence (all tokens except the last)
y = input_sequences[:, -1]   # Target word (last token)
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# =============================================================================
# 2. Build the LSTM-based Next-Word Prediction Model
# =============================================================================
st.header("Model Building")
st.write("Building LSTM-based next-word prediction model...")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=total_words, output_dim=100, input_length=(max_sequence_len - 1)),
    tf.keras.layers.LSTM(150, return_sequences=False),
    tf.keras.layers.Dense(total_words, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

st.subheader("Model Summary")
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
st.text("\n".join(model_summary))

# =============================================================================
# 3. Train the Model (Overnight) or Load Existing Checkpoint
# =============================================================================
st.header("Model Training / Checkpoint Loading")
epochs = 100  
batch_size = 64
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "lstm_best_model.h5")

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="accuracy",
    save_best_only=True,
    verbose=1,
)

if os.path.exists(checkpoint_path):
    st.success("Checkpoint found! Loading saved model weights...")
    model.load_weights(checkpoint_path)
    st.write("Model weights loaded. Skipping training.")
else:
    st.info("No checkpoint found. Training model from scratch...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    st.success("Training complete and checkpoint saved.")

# =============================================================================
# 4. Next-Word Prediction Function
# =============================================================================
def predict_next_words(seed_text, next_words=1):
    """
    Given a seed text, predict the next `next_words` tokens iteratively using the trained LSTM model.
    """
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=(max_sequence_len - 1), padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=1)[0]
        if predicted_index == 0:
            break
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                seed_text += " " + word
                break
    return seed_text

# =============================================================================
# 5. Streamlit Web Interface for Prediction
# =============================================================================
st.header("Next-Word Prediction")

col1, col2 = st.columns(2)
with col1:
    seed_text_input = st.text_input("Enter your seed text:", "The movie was")
with col2:
    num_words_input = st.number_input("Number of words to predict:", min_value=1, max_value=20, value=1, step=1)

if st.button("Predict"):
    with st.spinner("Generating predictions..."):
        prediction = predict_next_words(seed_text_input, int(num_words_input))
    st.success("Prediction generated!")
    st.markdown("### Prediction Output:")
    st.write(prediction)

st.markdown("---")
st.markdown("**Note:** This app uses an LSTM-based model for next-word prediction. If checkpoint weights are present, they will be loaded automatically, preventing the need for retraining.")
