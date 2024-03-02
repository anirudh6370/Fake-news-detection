import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
from data_preprocessing import load_data,preprocess_text

def train_models(df):
    x = df["text"]
    y = df["response"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    max_words = 10000  # Maximum number of words to consider
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(x_train)
    x_train_tokens = tokenizer.texts_to_sequences(x_train)
    x_test_tokens = tokenizer.texts_to_sequences(x_test)

    vectorization = TfidfVectorizer()
    x_train = vectorization.fit_transform(x_train)
    x_test = vectorization.transform(x_test)

    model_rf = RandomForestClassifier()
    model_rf.fit(x_train, y_train)
    dump(model_rf, 'model_rf.joblib')

    dump(vectorization, 'vectorization.joblib')
    dump(tokenizer, 'tokenizer.joblib')

    dump(tokenizer, 'tokenizer.joblib')

    # return model_lr, model_rf


df = load_data("artifacts\Fake.csv", "artifacts\True.csv")
df["text"] = df["text"].apply(preprocess_text)
train_models(df)

# Train the models
# model_lr, model_rf = train_models(df)

# Load the vectorizer and tokenizer
# vectorization = load('vectorization.joblib')
# tokenizer = load('tokenizer.joblib')
