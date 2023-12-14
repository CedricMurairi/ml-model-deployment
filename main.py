import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

english_max_sequence_length = 100
french_max_sequence_length = 100

with open('small_vocab_en', 'r') as f:
    data1 = f.read()

with open('small_vocab_fr', 'r') as f:
    data2 = f.read()

english_sentences = data1.split('\n')
french_sentences = data2.split('\n')


def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(
    english_sentences, french_sentences)

# Split the data into training and test sets
english_train, english_test, french_train, french_test = train_test_split(
    preproc_english_sentences, preproc_french_sentences, test_size=0.2, random_state=42)

# Load the model architecture from the JSON file
with open("model_final.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load the model weights from the HDF5 file
loaded_model.load_weights("model_final_weights.h5")


def translate_sentence_eng_to_fr(model, sentence, english_tokenizer, french_tokenizer):
    y_id_to_word = {value: key for key,
                    value in french_tokenizer.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = [english_tokenizer.word_index[word]
                for word in sentence.split()]
    sentence = pad_sequences(
        [sentence], maxlen=preproc_english_sentences.shape[-1], padding='post')
    sentences = np.array([sentence[0], preproc_english_sentences[0]])
    predictions = model.predict(sentences, len(sentences))

    # Handle the padding index (0)
    translated_sentence = ' '.join(
        [y_id_to_word[np.argmax(x)] for x in predictions[0]])

    # Filter out <PAD> tokens
    filtered_words = [
        word for word in translated_sentence.split() if word != '<PAD>']

    # Join the remaining words into a sentence
    trimmed_text = ' '.join(filtered_words)

    return trimmed_text


# Streamlit app
st.title("Language Translator")
st.subheader("English to French")

# English
try:
    english_text = st.text_area("English Text", "")

    if st.button("Translate", key="translate_eng"):
        if english_text:
            translated_french_text = translate_sentence_eng_to_fr(
                loaded_model, english_text.lower(), english_tokenizer, french_tokenizer)
            french_text = translated_french_text
            st.text_area("French Text", translated_french_text)
except Exception as e:
    st.error(f"Error: {e}")
