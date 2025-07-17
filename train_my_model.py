import os
import re
import json
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

n_epochs = 100
training_ratio = 0.9
max_seq_len = 10
embedding_dim = 100
lstm_units = 100
learning_rate = 0.01
batch_size = 32
model_dir = "model"
tfjs_export_dir = "lm_tfjs-data"
input_file_path = "data/train.txt"


def clean_text(text):
    """
    Entfernt Sonderzeichen, wandelt Text in Kleinbuchstaben um und normalisiert Leerzeichen.
    """
    text = text.lower()
    text = re.sub(r'[^a-zäöüß\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {input_file_path}")

with open(input_file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

cleaned_text = clean_text(raw_text)
corpus = cleaned_text.split()

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>", lower=True, filters='')
tokenizer.fit_on_texts([cleaned_text])
total_words = len(tokenizer.word_index) + 1
print(f"Vokabular-Größe: {total_words} Wörter")

input_sequences = []
for i in range(2, len(corpus) + 1):
    seq = tokenizer.texts_to_sequences([' '.join(corpus[:i])])[0]
    input_sequences.append(seq)

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

indices = np.arange(len(X))
np.random.shuffle(indices)
split_at = int(len(X) * training_ratio)
train_idx, test_idx = indices[:split_at], indices[split_at:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = Sequential([
    Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_seq_len - 1),
    LSTM(units=lstm_units, return_sequences=True),
    LSTM(units=lstm_units),
    Dense(units=total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001)
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=n_epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
perplexity = np.exp(test_loss)
print(f"\n📈 Testgenauigkeit: {test_acc * 100:.2f} %")
print(f"🔍 Perplexity: {perplexity:.2f}")

os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "model.h5"))

os.makedirs(tfjs_export_dir, exist_ok=True)
tfjs.converters.save_keras_model(model, tfjs_export_dir)

with open(os.path.join(model_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

with open(os.path.join(model_dir, 'tokenizer_word2index.json'), 'w') as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=2)

with open(os.path.join(model_dir, 'tokenizer_index2word.json'), 'w') as f:
    json.dump(tokenizer.index_word, f, ensure_ascii=False, indent=2)

print(f"\n✅ Tokenizer und Modell erfolgreich gespeichert.")