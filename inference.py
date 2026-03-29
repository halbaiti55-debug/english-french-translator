import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models
encoder_model = tf.keras.models.load_model("artifacts/encoder_model.keras", compile=False)
decoder_model = tf.keras.models.load_model("artifacts/decoder_model.keras", compile=False)
# Load tokenizers
with open("artifacts/en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("artifacts/fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load(f)

# Load config
with open("artifacts/config.pkl", "rb") as f:
    config = pickle.load(f)

max_en_len = config["max_en_len"]
max_fr_len = config["max_fr_len"]
start_id = config["start_id"]
end_id = config["end_id"]

# Reverse dictionary for French words
fr_index_word = {v: k for k, v in fr_tokenizer.word_index.items()}

def clean_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-zà-ÿ\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def translate_sentence(sentence: str) -> str:
    sentence = clean_text(sentence)
    if not sentence:
        return ""

    seq = en_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_en_len, padding="post", truncating="post")

    states = encoder_model.predict(seq, verbose=0)

    target_token = np.array([[start_id]], dtype="int32")
    decoded_words = []

    for _ in range(max_fr_len):
        output, h, c = decoder_model.predict([target_token] + states, verbose=0)
        sampled_id = int(np.argmax(output[0, 0, :]))

        if sampled_id == end_id or sampled_id == 0:
            break

        word = fr_index_word.get(sampled_id, "")
        if word:
            decoded_words.append(word)

        target_token = np.array([[sampled_id]], dtype="int32")
        states = [h, c]

    return " ".join(decoded_words)
