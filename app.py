import streamlit as st
import pandas as pd
import streamlit as st

# Set the path to your image file on your desktop
image_path = 'images/smartwatch.webp'

# Display the image using the path
st.image(image_path, caption='Your Image', use_column_width=True)

from model.model import model_streamlit
from model.preprocessing import test_preprocessor

st.title('Predicting Activity')

# Input fields for user to input data
watch_type = st.text_input('What is the brand of your watch? (apple/fitbit)')

        model, acc, label_encoder = model_streamlit(watch_type.lower(), data)

# Read user's CSV file as a DF
# https://kitt.lewagon.com/camps/1543/lectures/content/05-ML_01-Fundamentals-of-Machine-Learning.html
input_data = st.file_uploader(label='Upload health data CSV', type=['csv'])

if input_data is not None:
    data = pd.read_csv(input_data)
    st.write(data.head())
if input_data is not None:
    data = pd.read_csv(input_data)
    st.write(data.head())

    model, acc, label_encoder = model_streamlit(watch_type.lower(), data)
    model, acc, label_encoder = model_streamlit(watch_type.lower(), data)

    # Display the prediction result
    st.write('Watch Type:', watch_type.lower())
    # Display the prediction result
    st.write('Watch Type:', watch_type.lower())

    st.write('Model:', model)
    st.write('Model:', model)

    st.write('Accuracy:', acc)
    st.write('Accuracy:', acc)

    test_watch = st.text_input('Do you want to test your watch? (yes/no)')

    if test_watch.lower() == 'yes':

        input_test_data = st.file_uploader(label='Upload health TEST data CSV', type=['csv'])

        if input_test_data is not None:
            data_test = pd.read_csv(input_test_data)
            st.write(data_test.head())

            # preprocess the data_test
            data_test_preprocessed = test_preprocessor(watch_type.lower(), data_test)
            prediction = model.predict(data_test_preprocessed)
    test_watch = st.text_input('Do you want to test your watch? (yes/no)')

    if test_watch.lower() == 'yes':

        input_test_data = st.file_uploader(label='Upload health TEST data CSV', type=['csv'])

        if input_test_data is not None:
            data_test = pd.read_csv(input_test_data)
            st.write(data_test.head())

            # preprocess the data_test
            data_test_preprocessed = test_preprocessor(watch_type.lower(), data_test)
            prediction = model.predict(data_test_preprocessed)

            st.write('Prediction:', label_encoder.inverse_transform(prediction))
            st.write('Prediction:', label_encoder.inverse_transform(prediction))
