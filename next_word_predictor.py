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
- Tracks Top‑1 & Top‑5 accuracy, and Perplexity.
- Resumes from checkpoint if available.
""")
st.sidebar.info("Metrics are computed on a validation set (20% split).")

# =============================================================================
# Title and Description
# =============================================================================
st.title("Next Word Predictor (Embedding + LSTM)")
st.markdown("""
This app trains (or loads) an LSTM model for next‑word prediction and evaluates it on a held-out validation set,
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

tokenizer = Tokenizer()
tokenizer.fit_on_texts([full_text])
vocab_size = len(tokenizer.word_index) + 1
st.write(f"Vocabulary size: {vocab_size}")

tokens = tokenizer.texts_to_sequences([full_text])[0]
tokens = tokens[:20000]
st.write(f"Using {len(tokens)} tokens.")

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
st.write(f"Training: {len(X_train)} | Validation: {len(X_val)}")

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
# 3. Train or Load Checkpoint
# =============================================================================
st.header("3. Train / Load Checkpoint")

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(checkpoint_dir, "lstm_best_model.h5")
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

if os.path.exists(ckpt_path):
    st.success("✅ Loaded saved model from checkpoint.")
    model.load_weights(ckpt_path)
else:
    st.info("🚀 Training from scratch...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[ckpt_cb],
        verbose=1
    )
    st.success("🏁 Training complete and checkpoint saved.")

# =============================================================================
# 4. Evaluate and Show Metrics
# =============================================================================
st.header("4. Evaluation Metrics")

val_loss, val_acc, val_top5 = model.evaluate(X_val, y_val, verbose=0)
perplexity = math.exp(val_loss)

# Streamlit display
st.metric("Validation Loss", f"{val_loss:.4f}")
st.metric("Accuracy (Top‑1)", f"{val_acc*100:.2f}%")
st.metric("Top‑5 Accuracy", f"{val_top5*100:.2f}%")
st.metric("Perplexity", f"{perplexity:.2f}")

# Terminal printout
print("\n=== Evaluation Metrics ===")
print(f"Validation Loss   : {val_loss:.4f}")
print(f"Accuracy (Top‑1)  : {val_acc*100:.2f}%")
print(f"Top‑5 Accuracy    : {val_top5*100:.2f}%")
print(f"Perplexity        : {perplexity:.2f}")

# =============================================================================
# 5. Prediction Function
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
# 6. Streamlit UI
# =============================================================================
st.header("5. Next‑Word Prediction Demo")
col1, col2 = st.columns(2)
with col1:
    seed_text = st.text_input("Enter seed text:", "The movie was")
with col2:
    num_words = st.number_input("Words to predict:", min_value=1, max_value=20, value=1)

if st.button("Predict"):
    with st.spinner("Generating..."):
        result = predict_next(seed_text, num_words)
    st.success("✅ Prediction generated!")
    st.markdown("**Result:**")
    st.write(result)

st.markdown("---")
st.caption("ℹ️ Evaluation metrics shown above are from a held‑out validation set.")
