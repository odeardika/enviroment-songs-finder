import streamlit as st
import backend as rb
import joblib
import pandas as pd

model = joblib.load("train_model.pkl")
df = pd.read_csv("tfidf_result.csv").iloc[:,1:]

with st.sidebar:
    st.image('Logo.png',width=200)
    st.sidebar.title("About")
    st.write(
    "Aplikasi ini bertujuan untuk membantu menemukan lagu yang memiliki makna atau mengajarkan"+
    " untuk menjaga lingkungan dengan menggunakan lirik dari lagu tersebut."
        )
    st.write("Cara penggunaan:")
    st.write("1. Mencari lirik lagu yang ingin di cek 🛜")
    st.write("2. Memasukan lirik lagu tersebut pada kolom yang tersedia 📝")
    st.write("3. Tekan tombol Classify untuk melihat hasilnya 📊")
st.image('Title.png')
st.header("Masukan Songs Lyrics :")
lyrics = st.text_area('',value='')
if st.button("Classify 🔍"):
    input = rb.preprocess(lyrics)
    result = rb.check_emotion(df, model, input)
    if result == "lingkungan":
        st.header(":red[Lirik Lagu ini tidak ada unsur menjaga lingkungan]")
    else:
        st.header(":green[Lirik Lagu ini ada unsur menjaga lingkungan]")