import streamlit as st
import pandas as pd
import plotly.express as px

def run():
    # load (original) data
    df = pd.read_csv('weather.csv', usecols=['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow'])

    # constants
    MIN_DATE = pd.to_datetime(df['Date']).min().date()
    MAX_DATE = pd.to_datetime(df['Date']).max().date()

    # pie chart
    def plot_pie_chart(df, column_name):
        column_count = df[column_name].value_counts()
        fig = px.pie(names=column_count.index, values=column_count.values, title=f'Distribution of {column_name}')
        st.plotly_chart(fig)

    st.write("# Visualise data")
    view_option = st.radio('Want do you want to see?', options=['Data overview', 'Histograms', 'Data for selected date', 'Maximum and minimum temperature', 'Line charts'])

    if view_option == 'Data overview':
        st.write("The original dataset contains", df.shape[0], 'rows and', df.shape[1], 'columns.') 
        st.write("Here you can see a panoramic of the first rows:")
        st.write(df.head())
        st.write("Here you can see a pie chart of some data:")
        # Show pie chart for selected column
        column_option = st.selectbox('Select a column to show distribution', options=['Location', 'RainToday', 'RainTomorrow'])
        plot_pie_chart(df, column_option)
    elif view_option == 'Histograms':
        # Show histogram for selected column
        column_option = st.selectbox('Select a column to show distribution', options=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm'])
        fig = px.histogram(df, x=column_option, nbins=20, title=f'Distribution of {column_option}')
        st.plotly_chart(fig)
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
    elif view_option == 'Line charts':
        # Show line chart for selected column
        column_option = st.selectbox('Select a column to show trend', options=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm'])
        df_sorted = df.sort_values(by='Date')
        fig = px.line(df_sorted, x='Date', y=column_option, title=f'Trend of {column_option}')
        st.plotly_chart(fig)
