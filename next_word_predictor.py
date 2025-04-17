import os
import math
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
- LSTM + Embedding next‑word predictor on IMDB Reviews (subset).
- Tracks Top‑1 & Top‑5 accuracy, perplexity.
- Checkpoint loading to resume without retraining.
""")
st.sidebar.info("Metrics are computed on a held‑out validation set.")

# =============================================================================
# Main Title and Description
# =============================================================================
st.title("Next Word Predictor (Embedding + LSTM)")
st.markdown("""
This app trains (or loads) an LSTM model for next‑word prediction and **evaluates** it on a validation set,
reporting **loss**, **accuracy**, **top‑5 accuracy**, and **perplexity**.
""")

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
st.header("1. Data Loading & Preprocessing")

ds_train = tfds.load("imdb_reviews", split="train", shuffle_files=True)
text_list, max_examples = [], 2000
for i, ex in enumerate(tfds.as_numpy(ds_train)):
    text_list.append(ex['text'].decode('utf-8'))
    if i >= max_examples - 1:
        break
full_text = "\n".join(text_list)
st.write(f"Loaded {len(full_text)} characters.")

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([full_text])
vocab_size = len(tokenizer.word_index) + 1
st.write(f"Vocab size: {vocab_size}")

tokens = tokenizer.texts_to_sequences([full_text])[0]
tokens = tokens[:20000]
st.write(f"Using {len(tokens)} tokens.")

# Create sequences
max_len = 20
sequences = []
for i in range(max_len, len(tokens)):
    sequences.append(tokens[i-max_len:i+1])
sequences = np.array(sequences)
st.write(f"Total sequences: {sequences.shape[0]}")

X = sequences[:, :-1]
y = sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Train/Validation split (80/20)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
st.write(f"Training on {len(X_train)} seqs; validating on {len(X_val)} seqs.")

# =============================================================================
# 2. Build the Model
# =============================================================================
st.header("2. Model Building")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    ]
)
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
st.subheader("Model Summary")
st.text("\n".join(model_summary))

# =============================================================================
# 3. Train or Load Checkpoint & Evaluate
# =============================================================================
st.header("3. Training / Checkpoint & Evaluation")

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(checkpoint_dir, "lstm_best_model.h5")
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

if os.path.exists(ckpt_path):
    st.success("Loading saved weights from checkpoint.")
    model.load_weights(ckpt_path)
else:
    st.info("No checkpoint found. Training from scratch...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[ckpt_cb],
        verbose=1
    )
    st.success("Training complete and checkpoint saved.")

# Evaluate on validation set
val_loss, val_acc, val_top5 = model.evaluate(X_val, y_val, verbose=0)
perplexity = math.exp(val_loss)

# Display metrics
st.subheader("Validation Metrics")
st.metric("Loss", f"{val_loss:.4f}")
st.metric("Accuracy", f"{val_acc*100:.2f}%")
st.metric("Top‑5 Accuracy", f"{val_top5*100:.2f}%")
st.metric("Perplexity", f"{perplexity:.2f}")

# =============================================================================
# 4. Prediction Function
# =============================================================================
def predict_next(seed, n_words=1):
    text = seed
    for _ in range(n_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_len, padding="pre")
        probs = model.predict(seq, verbose=0)[0]
        idx = np.argmax(probs)
        if idx == 0: break
        word = tokenizer.index_word.get(idx, "")
        text += " " + word
    return text

# =============================================================================
# 5. Streamlit UI for Prediction
# =============================================================================
st.header("4. Next‑Word Prediction Demo")
c1, c2 = st.columns(2)
with c1:
    seed = st.text_input("Seed text", "The movie was")
with c2:
    n = st.number_input("Words to predict",_min_value=1,max_value=20,value=1)

if st.button("Predict"):
    with st.spinner("Generating..."):
        out = predict_next(seed, n)
    st.success("Done!")
    st.markdown("**Output:**")
    st.write(out)

st.markdown("---")
st.markdown("ℹ️ **Note:** Validation metrics shown above measure Top‑1 & Top‑5 accuracy and perplexity on a held‑out set.")
