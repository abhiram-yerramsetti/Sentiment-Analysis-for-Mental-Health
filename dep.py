import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier


# Load the pre-trained model
model = pickle.load(open(r"E:\innomatics\ml\Sentiment Analysis for Mental Health\model.pkl", 'rb'))


st.image(r"E:\innomatics\logo.png",width=200)
st.title("mental Health Sentiment Analysis ")

# Input email text
text = st.text_input("Enter what your are fealing:")



sentiment = model.predict([text])[0]

# Display the prediction when the button is pressed
if st.button('Submit'):
    st.write("The email is:", sentiment)
        




