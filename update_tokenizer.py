from tensorflow.keras.preprocessing.text import Tokenizer
import json
import os

# Paths
train_file = 'data/train.txt'
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Load training text
with open(train_file, 'r', encoding='utf-8') as f:
    texts = f.readlines()

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Save tokenizer JSON
with open(os.path.join(model_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

# Save word2index
with open(os.path.join(model_dir, 'tokenizer_word2index.json'), 'w', encoding='utf-8') as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False)

# Save index2word
index2word = {str(i): w for w, i in tokenizer.word_index.items()}
with open(os.path.join(model_dir, 'tokenizer_index2word.json'), 'w', encoding='utf-8') as f:
    json.dump(index2word, f, ensure_ascii=False)

print("✅ Tokenizer updated successfully.")
