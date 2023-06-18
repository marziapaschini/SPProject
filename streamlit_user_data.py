import streamlit as st
import model as md

def run():
    st.write("# Weather forecast")
    st.write("### Will it rain tomorrow?")
    st.write("### Insert your current data and find out.")

    # user parameters insertion
    MinTemp = st.number_input("Enter today's minimum temperature (°C):", format="%0.1f", help="Temperature in degrees Celsius")
    MaxTemp = st.number_input("Enter today's maximum temperature (°C):", format="%0.1f", help="Temperature in degrees Celsius")
    RainToday = st.checkbox("Did it rain today?", ["Yes", "No"])
    if RainToday:
        Rainfall = st.number_input("Enter the amount of rain today (mm):", format="%0.1f", help="Rainfall amount in millimeters")
    else:
        Rainfall = 0
    Humidity = st.number_input("Enter today's humidity (%):", format="%0.1f", help="Humidity percentage")
    WindSpeed = st.number_input("Enter today's wind speed (km/h):", format="%0.1f", help="Wind speed in kilometers per hour")

    # data validation
    if MinTemp >= MaxTemp:
        st.error("Minimum temperature must be lower than maximum temperature")
    if not (0 <= Humidity <= 100):
        st.error("Humidity must be between 0% and 100%")
    if RainToday == "Yes":
        RainToday = True
    else:
        RainToday = False
    if Rainfall < 0:
        st.warning("Rainfall should be a positive value")
    if WindSpeed < 0:
        st.warning("Wind speed should be a positive value")

    # prediction
    if st.button("Generate forecast"):
        prediction = md.model.predict([[MinTemp, MaxTemp, Rainfall, Humidity, WindSpeed]])
        if prediction == 1:
            st.write("Yes, it might rain tomorrow.")
        else:
            st.write("No, it's not likely to rain tomorrow.")
