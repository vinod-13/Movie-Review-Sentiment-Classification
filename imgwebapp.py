import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("IMDB Movie Review Sentiment Classification")
st.text("Enter movie review")


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
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
path = st.text_input('Enter Text to Classify:','I can watch this movie forever just because of the beauty in its cinematography')
if path is not None:
    st.write("Input Text")
    st.write(path)
    st.write("Predicted Class :")
    with st.spinner('classifying.....'):

        validation_sentence = path
        validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
        validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                                   truncating='post', padding='post')
        #print(validation_sentence[0])
        #print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))


        #label =np.argmax(model.predict(decode_img(content)),axis=1)
        label = model.predict(validation_sentence_padded)[0]
        #st.write(classes[label[0]])
        if label > 0.5:
            st.write(classes[0])
        else:
            st.write(classes[1])
    st.write("")
    #image = Image.open(BytesIO(content))
    #st.image(image, caption='Classifying Bean Image', use_column_width=True)
