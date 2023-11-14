import streamlit as st
import pickle as pkl
from datetime import datetime

startTime = datetime.now()

filename = "model.h5"
model = pkl.load(open(filename, 'rb'))

sex_d = {0: "Kobieta", 1: "Mężczyzna"}
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}


def main():
    st.set_page_config(page_title="Czy przeżyłbyś katastrofę?",
                       page_icon=":ship:", layout="wide")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG%22)")

    with overview:
        st.title("Czy przeżyłbyś katastrofę?")

    with left:
        sex_radio = st.radio("Płeć:", list(
            sex_d.keys(), format_func=lambda x: sex_d[x]))
        pclass_radio = st.radio("Klasa:", list(
            pclass_d.keys(), format_func=lambda x: pclass_d[x]))
        embarked_radio = st.radio("Port:", list(
            embarked_d.keys(), index=2, format_func=lambda x: embarked_d[x]))

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=0, max_value=100)
        sibsp_slider = st.slider(
            "# Liczba rodzeństwa/partnerów", min_value=0, max_value=8)
        parch_slider = st.slider(
            "# Liczba rodziców/dzieci", min_value=0, max_value=6)
        fare_slider = st.slider(
            "Cena biletu", min_value=0, max_value=500, step=10)

    data = [startTime, filename, model, sex_d, pclass_d, embarked_d] #uzupelnic wszystkie zmienne potrzebne do modelu. Podac kolumny/cechy, do wytrenowania modelu
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.header("czy dana osoba przeżyje? {0}".format("Tak" if survival else "Nie")) #mozna dac survival == 1
        st.subheader("Pewność predyjchu: {0:2f} %".format(s_confidence[0][survival][0]*100))

if __name__ == "__main__":
    main()