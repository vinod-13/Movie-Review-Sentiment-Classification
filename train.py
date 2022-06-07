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

import warnings

warnings.filterwarnings('ignore')
sns.set()

tf.config.set_visible_devices([], 'GPU')

if tf.test.gpu_device_name():

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:
    print("Please install GPU version of TF")


def train():
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

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model = Sequential()

    model.add(Embedding(input_dim=num_words, output_dim=100,
                        input_length=128, trainable=True))
    model.add(LSTM(100, dropout=0.1, return_sequences=True))
    model.add(LSTM(100, dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test), callbacks=[callback])

    print("Model Accuracy is:{}".format(history.history['accuracy'][np.argmin(history.history['loss'])]))

    model.save('models/', save_format='tf')

    print("Predicting on newly trained and loaded model")

    validation_sentence = [
        'This movie was not good at all. It had some good parts like the acting was pretty good but the story was not impressing at all.']
    validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
    validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                               truncating='post', padding='post')
    print(validation_sentence[0])
    print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))

    validation_sentence = [
        'It had some bad parts like the storyline although the actors performed really well and that is wht overall I enjoyed it']
    validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
    validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                               truncating='post', padding='post')
    print(validation_sentence[0])
    print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))

    validation_sentence = ['I can watch this movie forever just because of the beauty in its cinematography']
    validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
    validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                               truncating='post', padding='post')
    print(validation_sentence[0])
    print("Probability of Positive: {}".format(model.predict(validation_sentence_padded)[0]))



    model_loaded = tf.keras.models.load_model('models/')

    validation_sentence = ['The move was such waste useless crap movie']
    validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
    validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128,
                                               truncating='post', padding='post')
    print(validation_sentence[0])
    print("Probability of Positive: {}".format(model_loaded.predict(validation_sentence_padded)[0]))

if __name__ == '__main__':
    train()