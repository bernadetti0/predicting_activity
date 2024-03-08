import streamlit as st

from model.model import model_streamlit

st.title('Predicting Activity')

# Input fields for user to input data
watch_type = st.text_input('Enter input data:')

model = model_streamlit(watch_type.lower())

# [['device', 'participant_id','age', 'gender', 'height','weight', 'steps', 'heart_rate', 'calories', 'distance', 'bmi']]
# [['apple',  '82093840928',    '30',  'male',   '173',   '90',    '123234',   '100',       '12311',     '20',   '56789']]

# Read user's CSV file as a DF
# https://kitt.lewagon.com/camps/1543/lectures/content/05-ML_01-Fundamentals-of-Machine-Learning.html
input_data = st.file_uploader(label='Upload health data CSV', type=['csv'])
# prediction = model.predict(input_data)
prediction = 'running'

# Display the prediction result
st.write('Prediction:', prediction)
st.write('Watch Type:', watch_type.lower())
