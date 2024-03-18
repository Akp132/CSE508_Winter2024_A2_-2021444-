import numpy as np
import pickle
from scipy.spatial.distance import cdist
import os
import pandas as pd

# Setting paths and loading dataset
results_directory = '/Users/akshay/Desktop/IR/Assinment2/output'
data_csv_path = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
data_frame = pd.read_csv(data_csv_path)

# Load retrieval results
results_file_path = os.path.join(results_directory, 'retrieval_results_new.pkl')
with open(results_file_path, 'rb') as results_file:
    retrieval_data = pickle.load(results_file)

# Unpack retrieval results
image_indices_matched = retrieval_data['similar_image_indices']
similarities_images = retrieval_data['image_similarities']
review_indices_matched = retrieval_data['similar_review_indices']
similarities_reviews = retrieval_data['review_similarities']

def compute_overall_scores(img_similarities, txt_similarities):
    """Calculate average scores from image and text similarities."""
    overall_scores = [(img_sim + txt_sim) / 2 for img_sim, txt_sim in zip(img_similarities, txt_similarities)]
    return overall_scores

# Calculate composite scores
overall_scores = compute_overall_scores(similarities_images, similarities_reviews)

# Sort based on composite score
sorted_results = sorted(zip(overall_scores, image_indices_matched, review_indices_matched), reverse=True, key=lambda pair: pair[0])

print("Top Matches Across Images and Reviews:")
for position, (score, img_index, txt_index) in enumerate(sorted_results, 1):
    print(f"Position: {position}, Img ID: {img_index}, Review ID: {txt_index}, Score: {score:.4f}")

# Save sorted results
sorted_results_file = os.path.join(results_directory, 'sorted_overall_results.pkl')
with open(sorted_results_file, 'wb') as file:
    pickle.dump(sorted_results, file)

print(f"Sorted matches saved to: {sorted_results_file}\n")

def fetch_items(data, img_idxs, review_idxs):
    """Retrieve specific items based on their indices."""
    highest_index = data.index.max()
    valid_img_idxs = [index for index in img_idxs if index <= highest_index]
    valid_review_idxs = [index for index in review_idxs if index <= highest_index]

    if not valid_img_idxs or not valid_review_idxs:
        print("Warning: Some indices exceed the data boundaries.")
        return [], []

    img_urls = data.loc[valid_img_idxs, 'Image URL'].tolist()
    review_texts = data.loc[valid_review_idxs, 'Review Text'].tolist()
    return img_urls, review_texts
