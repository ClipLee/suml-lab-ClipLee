import numpy as np
import streamlit as st
import pickle as pkl
from datetime import datetime

startTime = datetime.now()

filename = "Lab06/model.h5"
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

    st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

    with overview:
        st.title("Czy przeżyłbyś katastrofę?")

    with left:
        sex_radio = st.radio("Płeć:", list(
            sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa:", list(
            pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio("Port:", list(  # czy embarked ma tu sens? nie ma go w modelu
            embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=0, max_value=100)
        sibsp_slider = st.slider(
            "# Liczba rodzeństwa/partnerów", min_value=0, max_value=8)
        parch_slider = st.slider(
            "# Liczba rodziców/dzieci", min_value=0, max_value=6)
        fare_slider = st.slider(
            "Cena biletu", min_value=0, max_value=500, step=10)

    data = np.array([pclass_radio, age_slider, sibsp_slider,
                    parch_slider, fare_slider, sex_radio])

    survival = model.predict([data])
    s_confidence = model.predict_proba([data])

    with prediction:
        if survival:
            st.markdown(
                "<h2> Czy dana osoba przeżyje?: <span style='color: green;'>TAK</span></h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2>Czy dana osoba przeżyje?: <span style='color: red;'>NIE</span></h2>",
                        unsafe_allow_html=True)  # mozna dac survival == 1

        # st.subheader("Pewność predyktu: {0:2f} %".format(
        #     s_confidence[0][survival][0]*100))

        confidence = s_confidence[0][survival][0]*100
        # HSL color: Hue, Saturation, Lightness
        color = f'hsl({120 * confidence / 100}, 100%, 50%)'
        st.markdown(
            f"<h2 style='color: {color};'>Pewność predykcji: {confidence:.2f} %</h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
