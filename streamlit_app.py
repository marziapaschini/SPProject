import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit_user_data as stud
import streamlit_visualise_data as stvd

# Set page width and layout
st.set_page_config(
    page_title="UmbrellaAlert",
    page_icon=":umbrella_with_rain_drops:",
    layout="wide"
)

def homepage():
    st.title("☔UmbrellaAlert: an Australian Weather Forecast App")
    st.write(""" 
    <style>
    ul {
        font-size: 24px;
    }
    </style>
    
    ### **Welcome to UmbrellaAlert**, an application for weather forecasting in Australia.
    # 
    #### With UmbrellaAlert, you can:
    <ul>
        <li>Access detailed information about current and future weather conditions, including temperature, atmospheric pressure, chance of rain, and much more.</li>
        <li>View historical weather data in detail.</li>
        <li>Use current data to obtain personalized forecasts for the chance of rain for the following day.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.write("#### This can help you plan your day effectively. Choose one of the navigation options on the left to get started!")
         
# weather forecast page
def user_data_page():
    stud.run()

# visualising data page
def visualise_data_page():
    stvd.run()

# sidebar for navigation
menu = ["Homepage", "Weather Forecast", "Visualise Previous Data"]
choice = st.sidebar.selectbox("Choose a section", menu)

# sections navigation
if choice == "Homepage":
    homepage()
elif choice == "Weather Forecast":
    user_data_page()
elif choice == "Visualise Previous Data":
    visualise_data_page()