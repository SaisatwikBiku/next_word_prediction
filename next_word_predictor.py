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
- LSTM + Embedding next‑word predictor on IMDB Reviews.
- Tracks Top‑1 & Top‑5 accuracy, and Perplexity.
- Resumes from checkpoint if available.
""")
st.sidebar.info("Metrics are computed on a validation set (20% split).")

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
ds_train = tfds.load("imdb_reviews", split="train", shuffle_files=True)
text_list, max_examples = [], 2000
for i, ex in enumerate(tfds.as_numpy(ds_train)):
    text_list.append(ex['text'].decode('utf-8'))
    if i >= max_examples - 1:
        break
full_text = "\n".join(text_list)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([full_text])
vocab_size = len(tokenizer.word_index) + 1

tokens = tokenizer.texts_to_sequences([full_text])[0]
tokens = tokens[:20000]

max_len = 20
sequences = []
for i in range(max_len, len(tokens)):
    sequences.append(tokens[i-max_len:i+1])
sequences = np.array(sequences)

X = sequences[:, :-1]
y = sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# =============================================================================
# 2. Build the Model
# =============================================================================
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

# =============================================================================
# 3. Train or Load Checkpoint
# =============================================================================
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
    model.load_weights(ckpt_path)
else:
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[ckpt_cb],
        verbose=1
    )

# =============================================================================
# 4. Evaluate and Show Metrics
# =============================================================================
val_loss, val_acc, val_top5 = model.evaluate(X_val, y_val, verbose=0)
perplexity = math.exp(val_loss)

# Sidebar summary
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.write(f"Total Characters: {len(full_text)}")
st.sidebar.write(f"Vocabulary Size: {vocab_size}")
st.sidebar.write(f"Training Sequences: {len(X_train)}")
st.sidebar.write(f"Validation Sequences: {len(X_val)}")

st.sidebar.subheader("📈 Evaluation Metrics")
st.sidebar.metric("Loss", f"{val_loss:.4f}")
st.sidebar.metric("Top-1 Accuracy", f"{val_acc*100:.2f}%")
st.sidebar.metric("Top-5 Accuracy", f"{val_top5*100:.2f}%")
st.sidebar.metric("Perplexity", f"{perplexity:.2f}")

with st.sidebar.expander("🧠 Model Summary"):
    st.text("\n".join(model_summary))

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
st.header("Next‑Word Prediction Demo")

with st.form("prediction_form", clear_on_submit=False):
    seed_text = st.text_input("Seed Text", value="The movie was")
    num_words = st.slider("Number of words to predict", 1, 20, 3)
    submit = st.form_submit_button("Predict")

if submit:
    with st.spinner("Generating prediction..."):
        result = predict_next(seed_text, num_words)
    st.success("Prediction generated")
    st.markdown("### Predicted Text")
    st.code(result, language="markdown")
