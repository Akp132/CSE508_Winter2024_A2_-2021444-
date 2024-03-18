import numpy as np
import pandas as pd
import pickle
import os

# Define directories and load the dataset
data_directory = '/Users/akshay/Desktop/IR/Assinment2/output'
data_file_path = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
analysis_data = pd.read_csv(data_file_path)

# Load retrieval results from file
retrieval_file_path = os.path.join(data_directory, 'retrieval_results_new.pkl')
with open(retrieval_file_path, 'rb') as retrieval_file:
    retrieval_data = pickle.load(retrieval_file)

# Unpacking retrieval data
matched_image_idxs = retrieval_data['similar_image_indices']
image_match_scores = retrieval_data['image_similarities']
matched_review_idxs = retrieval_data['similar_review_indices']
review_match_scores = retrieval_data['review_similarities']

def compute_aggregate_scores(img_scores, txt_scores):
    """Calculate average scores for image and text similarities."""
    aggregate_scores = sorted([(idx, (img + txt) / 2) for idx, (img, txt) in enumerate(zip(img_scores, txt_scores))], key=lambda pair: pair[1], reverse=True)
    return aggregate_scores

# Generate aggregate scores
aggregate_scores = compute_aggregate_scores(image_match_scores, review_match_scores)

def fetch_content_by_index(content_frame, selected_indices):
    """Retrieve image URLs and reviews based on provided indices."""
    filtered_indices = [i for i in selected_indices if i < len(content_frame)]
    
    if not filtered_indices:
        return [], []

    urls = content_frame.iloc[filtered_indices]['Image'].tolist()
    review_content = content_frame.iloc[filtered_indices]['Review Text'].tolist()
    return urls, review_content

def showcase_results(content_frame, agg_scores, modified_reviews=None):
    """Displaying the retrieval results."""
    print("IMAGE & TEXT MATCHES:")
    for position, (index, agg_score) in enumerate(agg_scores, 1):
        if index < len(content_frame):
            img_url = content_frame.iloc[index]['Image'] if 'Image' in content_frame.columns else "No image URL"
            review_text = content_frame.iloc[index]['Review Text'] if 'Review Text' in content_frame.columns else "No review"
            print(f"{position}) Img URL: {img_url}, Review: {review_text}, Img Score: {image_match_scores[index]:.4f}, Txt Score: {review_match_scores[index]:.4f}")
        else:
            print(f"Rank {position} skipped, index {index} is beyond range.")

    # Optionally handle modified reviews
    if modified_reviews:
        print("REVISED TEXT RETRIEVAL:")
        # This section could be similar to the above, tailored for modified reviews

# Invoke display function
showcase_results(analysis_data, aggregate_scores)

# Summarize and report on composite scores
image_score_average = np.mean(image_match_scores)
text_score_average = np.mean(review_match_scores)
combined_average_score = (image_score_average + text_score_average) / 2

print(f"Average Image Score: {image_score_average}")
print(f"Average Text Score: {text_score_average}")
print(f"Overall Average Score: {combined_average_score}")
