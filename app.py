""" Main functionality """
import csv
import os
from predicition import predict
import streamlit as st

ROOT = os.getcwd()

with open(ROOT+'\\data\\glnames.csv', mode='r') as infile:
    reader = csv.reader(infile)
    gl_names_dict = {rows[0]:rows[1] for rows in reader}

st.title('What GL Acct?')
st.markdown('Which GL do your words sounds like?')

st.header("Enter some words")
col1, col2 = st.columns(2)

with col1:
    # st.text("Words for prediction")
    words_to_predict = st.text_input('', '')

if st.button("Predict!"):
    result = predict(words_to_predict)
    st.text(f'{result[0]} : {gl_names_dict.get(result[0],"No GL Name")}')