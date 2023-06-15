import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# homepage
def homepage():
    st.title("Australian Weather Forecast App")
    # homepage content
         
# weather forecast page
def user_data_page():
    import streamlit_user_data as stud
    stud.run()

# visualising data page
def visualise_data_page():
    import streamlit_visualise_data as stvd
    stvd.run()

# sidebar for navigation
menu = ["Homepage", "Weather Forecast", "Visualise Previous Data"]
choice = st.sidebar.selectbox("Scegli una sezione", menu)

# sections navigation
if choice == "Homepage":
    homepage()
elif choice == "Weather Forecast":
    user_data_page()
elif choice == "Visualise Previous Data":
    visualise_data_page()
