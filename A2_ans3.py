import numpy as np
import pandas as pd
import pickle
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
from scipy.spatial.distance import cosine
import os
import torch
from torchvision import models, transforms
import requests
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from collections import defaultdict, Counter
from math import log
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re

# Define the path to your dataset and precomputed features
dataset_path = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
output_dir = '/Users/akshay/Desktop/IR/Assinment2/output'
image_features_path = os.path.join(output_dir, 'image_features.pkl')
tf_idf_scores_path = os.path.join(output_dir, 'tf_idf_scores.pkl')

# Load the dataset
df = pd.read_csv(dataset_path)

# Load precomputed image features and TF-IDF scores
with open(image_features_path, 'rb') as f:
    image_features = pickle.load(f)
with open(tf_idf_scores_path, 'rb') as f:
    tf_idf_scores = pickle.load(f)

resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()  # Set the model to evaluation mode

# Define image transformations
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def cosine_similarity(v1, v2):

  if isinstance(v1, np.ndarray) and all(isinstance(v, np.ndarray) for v in v2):
          # Convert list of numpy arrays (v2) to a single 2D numpy array
          v2 = np.array(v2)
          # Normalize v1 and v2
          v1_norm = v1 / np.linalg.norm(v1)
          v2_norm = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
          # Calculate cosine similarity
          similarities = np.dot(v1_norm, v2_norm.T)

      # Case for sparse vectors (TF-IDF scores)
  elif isinstance(v1, dict) and all(isinstance(v, dict) for v in v2):
          similarities = []
          for tfidf_dict in v2:
              # Intersection of keys (terms present in both vectors)
              common_terms = set(v1.keys()) & set(tfidf_dict.keys())
              # Manual dot product for common terms
              dot_product = sum(v1[term] * tfidf_dict[term] for term in common_terms)
              # Norms of the vectors
              norm_v1 = np.sqrt(sum(value ** 2 for value in v1.values()))
              norm_v2 = np.sqrt(sum(value ** 2 for value in tfidf_dict.values()))
              # Cosine similarity
              if norm_v1 == 0 or norm_v2 == 0:
                  similarity = 0
              else:
                  similarity = dot_product / (norm_v1 * norm_v2)
              similarities.append(similarity)
          similarities = np.array(similarities)

  else:
          raise ValueError("Unsupported input types.")

  return similarities



def find_most_similar_images(processed_image, precomputed_features, top_n=3):
    # Assuming direct comparison of processed_image array to precomputed feature vectors
    similarities = cosine_similarity(processed_image, precomputed_features).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_indices, [similarities[i] for i in top_indices]

def find_most_similar_reviews(input_tfidf, precomputed_tfidf_scores, top_n=3):
    # Calculate cosine similarity between the input TF-IDF vector and each precomputed TF-IDF vector
    similarities = cosine_similarity(input_tfidf, precomputed_tfidf_scores).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_similarities = [similarities[i] for i in top_indices]
    return top_indices, top_similarities

def preprocess_text(text):
    """Basic text preprocessing"""
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize by splitting on whitespace
    # Optionally remove stopwords here
    return tokens
def compute_tf(tokenized_review):
    tf = {}
    for word in tokenized_review:
        tf[word] = tf.get(word, 0) + 1

    # Normalize term frequencies by the total number of words in the document
    total_words = len(tokenized_review)
    tf = {word: count / total_words for word, count in tf.items()}

    return tf

def preprocess_image(image_url):
    """Fetch and preprocess an image from a URL, then extract features."""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        # Apply preprocessing transformations
        processed_image = transform_pipeline(image).unsqueeze(0)  # Add batch dimension

        # Extract features with the model
        with torch.no_grad():
            features = resnet_model(processed_image)

        # Convert features to a numpy array
        features_np = features.numpy().flatten()
        return features_np
    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None
## INPUT ##
input_image_url = input("Enter the image URL: ")
input_review_text = input("Enter the review text: ")


# Preprocess review text
processed_tokens = preprocess_text(input_review_text)
input_review_tfidf = compute_tf(processed_tokens)

# Preprocess the image from URL
processed_image = preprocess_image(input_image_url)

if processed_image is not None:

    # Find the most similar images and reviews
    similar_image_indices, image_similarities = find_most_similar_images(processed_image, image_features)
    similar_review_indices, review_similarities = find_most_similar_reviews(input_review_tfidf, tf_idf_scores)

    print("Similar Image Indices:", similar_image_indices)
    print("Image Similarities:", image_similarities)
    print("Similar Review Indices:", similar_review_indices)
    print("Review Similarities:", review_similarities)
else:
    print("The specified image URL and review were not found in the dataset, or image processing failed.")

# Save the retrieval results
retrieval_results = {
    'similar_image_indices': similar_image_indices,
    'image_similarities': image_similarities,
    'similar_review_indices': similar_review_indices,
    'review_similarities': review_similarities,
}
results_path = os.path.join(output_dir, 'retrieval_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(retrieval_results, f)
