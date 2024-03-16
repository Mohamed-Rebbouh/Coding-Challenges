import streamlit as st 
from utils import import_model
st.write('Hate Speech and Offensive Language Detection')
spreech=st.text_input('insert your text')
but=st.button('Detect!!')
if but and spreech:
    pred=import_model(spreech)
    if pred == 0:
        st.write("Predicted Class: Hate Speech")
    elif pred == 1:
        st.write("Predicted Class: Offensive Language")
    elif pred == 2:
        st.write("Predicted Class: Neither")

