import streamlit as st
from joblib import load
from data_preprocessing import preprocess_text

# Streamlit app
st.title("Fake News Detection App")

# Input text box
text_input = st.text_area("Enter text to check for fake news:", "")
vectorization = load('vectorization.joblib')
tokenizer = load('tokenizer.joblib')
model_rf = load('model_rf.joblib')

# Predict button
if st.button("Predict"):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text_input)
    
    # Tokenize the input text
    input_tokens = tokenizer.texts_to_sequences([preprocessed_text])
    
    # Vectorize the input text
    input_vectorized = vectorization.transform([preprocessed_text])
    
    # Predict using Random Forest model
    prediction_rf = model_rf.predict(input_vectorized)
    
    # Display the prediction
    
    if prediction_rf[0] == 1:
        st.write("Random Forest Model Prediction: Fake News")
    else:
        st.write("Random Forest Model Prediction: True News")
