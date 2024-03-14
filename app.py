import streamlit as st
import pandas as pd
import streamlit as st
from model.model import model_streamlit
from model.preprocessing import test_preprocessor

# Set the path to your image file on your desktop
image_path = 'images/smartwatch.webp'

# Display the image using the path
st.image(image_path, caption='Your Image', use_column_width=True)

st.title('Predicting Activity')

# Input fields for user to input data
watch_type = st.text_input('What is the brand of your watch? (apple/fitbit)')
if watch_type.lower() == 'apple' or 'fitbit':
    input_data = st.file_uploader(label='Upload health data CSV', type=['csv'])

    if input_data is not None:
        data = pd.read_csv(input_data)
        st.write(data.head())

        model, acc, label_encoder = model_streamlit(watch_type.lower(), data)

        # Display the prediction result
        st.write('Watch Type:', watch_type.lower())

        #st.write('Model:', model)

        #st.write('Accuracy:', acc)

        # preprocess the data
        data_test_preprocessed = test_preprocessor(watch_type.lower(), data)
        prediction = model.predict(data_test_preprocessed)

        st.write('Prediction:', label_encoder.inverse_transform(prediction))
