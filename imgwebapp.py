import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Bean Image Classifier")
st.text("Provide URL of bean Image for image classification")

#Preprocessing code

imdb = pd.read_csv("IMDB Dataset.csv")
imdb.head()

imdb.sentiment.value_counts()

text = imdb['review'][0]
print(text)
print("<================>")
print(word_tokenize(text))

corpus = []
for text in imdb['review']:
    words = [word.lower() for word in word_tokenize(text)]
    corpus.append(words)

num_words = len(corpus)
print(num_words)

imdb.shape

train_size = int(imdb.shape[0] * 0.8)
X_train = imdb.review[:train_size]
y_train = imdb.sentiment[:train_size]

X_test = imdb.review[train_size:]
y_test = imdb.sentiment[train_size:]

tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=128, truncating='post', padding='post')

X_train[0], len(X_train[0])

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=128, truncating='post', padding='post')

X_test[0], len(X_test[0])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

#Preprocessing code

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes = ['positive', 'negative']

#def decode_img(image):
#  img = tf.image.decode_jpeg(image, channels=3)
#  img = tf.image.resize(img,[224,224])
#  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Text to Classify.. ','I can watch this movie forever just because of the beauty in its cinematography')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):

        validation_sentence = content
        validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
        validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                                   truncating='post', padding='post')
        print(validation_sentence[0])
        print("Probability of Positive: {}".format(model_loaded.predict(validation_sentence_padded)[0]))


        #label =np.argmax(model.predict(decode_img(content)),axis=1)
        label = model.predict(validation_sentence_padded)[0]
        #st.write(classes[label[0]])
        if label > 0.5:
            st.write(classes[0])
        else:
            st.write(classes[1])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Bean Image', use_column_width=True)
