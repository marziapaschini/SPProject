import streamlit as st
import pandas as pd

df = pd.read_csv('weather.csv', usecols=['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow'])

MIN_DATE = pd.to_datetime(df['Date']).min().date()
MAX_DATE = pd.to_datetime(df['Date']).max().date()

st.write("### Visualise data")
view_option = st.radio('Want do you want to see?', options=['Data overview', 'Data for selected date', 'Maximum and minimum temperature'])

if view_option == 'Data overview':
    st.write("The original dataset contains", df.shape[0], 'rows and', df.shape[1], 'columns.') 
    st.write("Here you can see a panoramic of the first rows:")
    st.write(df.head())
elif view_option == 'Data for selected date':
    selected_date = st.date_input('Select a date', min_value=MIN_DATE, max_value=MAX_DATE, value=MIN_DATE)
    df_selected_date = df[df['Date'] == str(selected_date)]
    st.write('Information about selected date:')
    if not df_selected_date.empty:
        st.write(df_selected_date)
    else:
        st.write('No information available for the selected date.')
elif view_option == 'Maximum and minimum temperature':
    date_from = st.date_input('Select the start date', min_value=MIN_DATE, max_value=MAX_DATE, value=MIN_DATE)
    date_to = st.date_input('Select the end date', min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE)
    if date_from > date_to:
        st.error('Error: Start date must be before or equal to end date')
    else:
        df_selected = df[(df['Date'] >= str(date_from)) & (df['Date'] <= str(date_to))]
        if not df_selected.empty:
            min_temp = df_selected['MinTemp'].min()
            max_temp = df_selected['MaxTemp'].max()
            st.write(f"The **minimum temperature** recorded in the selected time interval is **{min_temp}°C**.")
            st.write(f"The **maximum temperature** recorded in the selected time interval is **{max_temp}°C**.")
        else:
            st.write('No information available for the selected time range.')

def visualize_data(df, data_type, date_from=None, date_to=None):
    if data_type == "temperature":
        if date_from is not None and date_to is not None:
            df_selected = df[(df['Date'] >= str(date_from)) & (df['Date'] <= str(date_to))]
            min_temp = df_selected['MinTemp'].min()
            max_temp = df_selected['MaxTemp'].max()
            st.write(f"The minimum temperature recorded in the selected time range is {min_temp} degrees Celsius.")
            st.write(f"The maximum temperature recorded in the selected time range is {max_temp} degrees Celsius.")
        else:
            st.write(f"The minimum temperature recorded is {df['MinTemp'].min()} degrees Celsius.")
            st.write(f"The maximum temperature recorded is {df['MaxTemp'].max()} degrees Celsius.")
    elif data_type == "precipitations":
        if date_from is not None and date_to is not None:
            df_selected = df[(df['Date'] >= str(date_from)) & (df['Date'] <= str(date_to))]
            rainfall_sum = df_selected['Rainfall'].sum()
            st.write(f"The total amount of rainfall recorded in the selected time range is {rainfall_sum} mm.")
        else:
            st.write(f"The total amount of rainfall recorded is {df['Rainfall'].sum()} mm.")
    elif data_type == "humidity":
        if date_from is not None and date_to is not None:
            df_selected = df[(df['Date'] >= str(date_from)) & (df['Date'] <= str(date_to))]
            hum9am_mean = df_selected['Humidity9am'].mean()
            hum3pm_mean = df_selected['Humidity3pm'].mean()
            st.write(f"The average humidity recorded at 9am in the selected time range is {hum9am_mean}%.")
            st.write(f"The average humidity recorded at 3pm in the selected time range is {hum3pm_mean}%.")
        else:
            st.write(f"The average humidity recorded at 9am is {df['Humidity9am'].mean()}%.")
            st.write(f"The average humidity recorded at 3pm is {df['Humidity3pm'].mean()}%.")
    elif data_type == "wind speed":
        if date_from is not None and date_to is not None:
            df_selected = df[(df['Date'] >= str(date_from)) & (df['Date'] <= str(date_to))]
            wind9am_mean = df_selected['WindSpeed9am'].mean()
            wind3pm_mean = df_selected['WindSpeed3pm'].mean()
            st.write(f"The average wind speed recorded at 9am in the selected time range is {wind9am_mean} km/h.")
            st.write(f"The average wind speed recorded at 3pm in the selected time range is {wind3pm_mean} km/h.")
        else:
            st.write(f"The average wind speed recorded at 9am is {df['WindSpeed9am'].mean()} km/h.")
            st.write(f"The average wind speed recorded at 3pm is {df['WindSpeed3pm'].mean()} km/h.")
    elif data_type == "atmospheric pressure":
        if date_from is not None and date_to is not None:
            df_selected = df[(df['Date'] >= str(date_from)) & (df['Date'] <= str(date_to))]
            press9am_mean = df_selected['Pressure9am'].mean()
            press3pm_mean = df_selected['Pressure3pm'].mean()
            st.write(f"The average atmospheric pressure recorded at 9am in the selected time range is {press9am_mean} hPa.")
            st.write(f"The average atmospheric pressure recorded at 3pm in the selected time range is {press3pm_mean} hPa.")
        else:
            st.write(f"ciao")