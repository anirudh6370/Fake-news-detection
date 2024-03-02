import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def load_data(fake_csv_path, true_csv_path):
    df1 = pd.read_csv(fake_csv_path)
    df2 = pd.read_csv(true_csv_path)
    df1["response"] = 1
    df2["response"] = 0
    df1.drop(["title", "subject", "date"], axis=1, inplace=True)
    df2.drop(["title", "subject", "date"], axis=1, inplace=True)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    df = merged_df.sample(frac=1, ignore_index=True)
    
    return df
