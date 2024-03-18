import os
import re
import math
import pickle
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter

# Ensure all necessary resources from NLTK are available
nltk.download('punkt')
nltk.download('stopwords')

def normalize_and_process_text(text):
    """Normalizes and processes the input text for further analysis."""
    # Convert text to lowercase and remove URLs and user mentions
    text = text.lower()
    text_without_urls = re.sub(r'\b(http\S+|@\w+)\b', '', text)
    
    # Tokenize the text, omitting punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text_without_urls)
    
    # Remove stopwords and apply stemming
    processed_tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    return processed_tokens

# Setup for text processing
stopwords_set = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Load and preprocess the dataset
dataset_filepath = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
reviews_df = pd.read_csv(dataset_filepath)
all_reviews = reviews_df['Review Text'].fillna(' ').apply(str).tolist()
processed_reviews = [normalize_and_process_text(review) for review in all_reviews]

def compute_tf_idf(doc_tokens_list):
    """Computes the TF-IDF scores for a list of documents."""
    # Calculate Term Frequency
    term_freq_list = [Counter(doc_tokens) for doc_tokens in doc_tokens_list]
    
    # Calculate Document Frequency and Inverse Document Frequency
    total_docs = len(doc_tokens_list)
    inverse_doc_freq = {}
    for doc in term_freq_list:
        for term in doc:
            inverse_doc_freq[term] = inverse_doc_freq.get(term, 0) + 1
    for term in inverse_doc_freq:
        inverse_doc_freq[term] = math.log(total_docs / inverse_doc_freq[term])
    
    # Compute TF-IDF
    tf_idf_scores = []
    for doc in term_freq_list:
        tf_idf_doc = {}
        for term, freq in doc.items():
            tf_idf_doc[term] = freq * inverse_doc_freq[term]
        tf_idf_scores.append(tf_idf_doc)

    return tf_idf_scores

tf_idf_results = compute_tf_idf(processed_reviews)

# Define directory for saving the results
output_directory = '/Users/akshay/Desktop/IR/Assinment2/output'
os.makedirs(output_directory, exist_ok=True)

# Save the processed text and TF-IDF scores
with open(os.path.join(output_directory, 'processed_reviews.pkl'), 'wb') as file:
    pickle.dump(processed_reviews, file)
with open(os.path.join(output_directory, 'tf_idf_scores.pkl'), 'wb') as file:
    pickle.dump(tf_idf_results, file)
