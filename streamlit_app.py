import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import checks
import work

df = pd.read_csv('weather.csv')

st.write("""
         # Australia **Weather Forecast**
         ### "Will it rain tomorrow?"
         ###### A web application for weather prediction
         """)

categoric_cols = checks.categoric_cols
numeric_cols = checks.numeric_cols
         
def view_previous_data():
    st.write("Original data:")
    #st.write(checks.numeric_cols)
    #st.write("Pre-processed data:")
    #st.write(work.X_numerical)
    #plot = checks.missing_categoric_data(df, categoric_cols)
    #st.pyplot(plot)

def insert_user_data():
    option = st.selectbox('Select the location:',df['Location'].unique())
    temp = st.text_input('Insert the temperature:', key='temp_input')
    if temp.isdigit():
        temp = round(float(temp), 1)
        st.write("Location:", option)
        st.write("Temperature:", temp)
    else:
        st.error("Please enter a valid temperature (a number with one decimal place).")
        return


# Crea due colonne per posizionare i pulsanti sulla stessa riga
col1, col2 = st.columns(2)

# Aggiungi pulsanti alle colonne
if col1.button('Visualize previous weather data'):
    st.subheader("Visualize previous weather data")
    view_previous_data()
if col2.button('Insert your data and get the prediction'):
    st.subheader("Insert your data and get the prediction")
    insert_user_data()