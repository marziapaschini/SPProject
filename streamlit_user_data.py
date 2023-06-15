import streamlit as st
import model as md

def run():
    st.write("# Weather forecast")

    # user parameters insertion
    MinTemp = st.number_input("Enter today's minimum temperature (°C):", format="%0.1f", help="Temperature in degrees Celsius")
    MaxTemp = st.number_input("Enter today's maximum temperature (°C):", format="%0.1f", help="Temperature in degrees Celsius")
    Rainfall = st.number_input("Enter the amount of rain today (mm):", format="%0.1f", help="Rainfall amount in millimeters")
    Humidity = st.number_input("Enter today's humidity (%):", format="%0.1f", help="Humidity percentage")
    WindSpeed = st.number_input("Enter today's wind speed (km/h):", format="%0.1f", help="Wind speed in kilometers per hour")

    # prediction
    if st.button("Generate forecast"):
        prediction = md.model.predict([[MinTemp, MaxTemp, Rainfall, Humidity, WindSpeed]])
        if prediction == 1:
            st.write("Yes, it might rain tomorrow.")
        else:
            st.write("No, it's not likely to rain tomorrow.")
