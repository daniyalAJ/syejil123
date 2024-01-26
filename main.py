# -------------------------- Preprocessing and utility functions -------------------------- #

# ------------- stopword and special character removal ------------- #
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import joblib
nltk.download('stopwords')
nltk.download('punkt')

stemmer = SnowballStemmer("german")

def preprocess_text(text):

    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    stop_words = set(stopwords.words('german'))
    tokens = [word for word in tokens if word not in stop_words]
    
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

# ------------- preprocessing data for tokenization  ------------- #

df = pd.read_csv('sample_data.csv')
df['input'] = df['text'].apply(preprocess_text)

# ------------- model loading  ------------- #

import tensorflow as tf
model = tf.keras.models.load_model('model.h5')

# -------------------------- Building API -------------------------- #

print("Ready")

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel): 
    input_string: str

@app.get("/")
def read_root():
    
    return {"message": "post your text as <input_string: your string>"}

@app.post("/predict/")
async def post_string(data: InputData):
    
    tokens = preprocess_text(data.input_string) #Stopword and special character removal
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['input'])
    sequence = tokenizer.texts_to_sequences([tokens]) #Tokenization
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post') # Padding
    X = np.array(padded_sequence) #Data transformation
    predictions = model.predict(X) #Prediction
    label_binarizer = joblib.load('label_binarizer.pkl') #load binarizer
    output = label_binarizer.inverse_transform(predictions) #Converting one-hot to original
    output = output[0] #Extracting final output

    return {"result": f"Class: {output}"}

