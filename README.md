# Fake News Detection

![Fake News Detection Banner](https://github.com/anirudh6370/Fake-news-detection/blob/main/fake%20news.png)

## Overview

This project aims to detect fake news using machine learning techniques. It uses a dataset containing news articles labeled as either fake or true, and trains machine learning models to classify new articles as either fake or true.

## Features

- Preprocesses text data to remove punctuation, convert to lowercase, and remove stopwords.
- Uses TF-IDF vectorization to convert text data to numerical features.
- Trains machine learning models (Logistic Regression and Random Forest) to classify news articles as fake or true.
- Provides a Streamlit app for interactive testing of the trained models.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anirudh6370/Fake-news-detection.git

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```
   
5. The application will launch in your default web browser, displaying the user interface.
