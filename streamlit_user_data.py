import streamlit as st

def insert_user_data():
    option = st.selectbox('Seleziona la posizione:', ['Milano', 'Roma', 'Napoli', 'Firenze'])
    temp = st.text_input('Inserisci la temperatura:', key='temp_input')
    if temp.isdigit():
        temp = round(float(temp), 1)
        st.write("Posizione:", option)
        st.write("Temperatura:", temp)
    else:
        st.error("Per favore inserisci una temperatura valida (un numero con una cifra decimale).")

st.write("### Insert data")
insert_user_data()