import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import math
import pickle
from collections import defaultdict

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
file_path = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
df = pd.read_csv(file_path)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess text data
df['preprocessed_text'] = df['Review Text'].fillna("").apply(preprocess_text)
text_data = df['preprocessed_text'].values

# Functions for TF-IDF calculation from scratch
def compute_tf(text):
    words = text.split()
    n = len(words)
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1
    tf = {word: count / n for word, count in word_counts.items()}
    return tf

def compute_idf(documents):
    N = len(documents)
    idf = defaultdict(lambda: 0)
    for document in documents:
        unique_words = set(document.split())
        for word in unique_words:
            idf[word] += 1
    idf = {word: math.log(N / count) for word, count in idf.items()}
    return idf

def compute_tfidf(tf, idf):
    tfidf = {word: tf_value * idf.get(word, 0) for word, tf_value in tf.items()}
    return tfidf

# Calculate TF for each document
tf = [compute_tf(text) for text in text_data]

# Calculate IDF for all documents
idf = compute_idf(text_data)

# Calculate TF-IDF for each document
tfidf = [compute_tfidf(doc_tf, idf) for doc_tf in tf]

# Save preprocessed text and TF-IDF scores using pickle
with open('/Users/akshay/Desktop/IR/Assinment2/output/preprocessed_text.pkl', 'wb') as f:
    pickle.dump(df['preprocessed_text'], f)

with open('/Users/akshay/Desktop/IR/Assinment2/output/tf_idf_scores.pkl', 'wb') as f:
    pickle.dump(tfidf, f)