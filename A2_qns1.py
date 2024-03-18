import torch
import os
from torchvision import models, transforms
import requests
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import pandas as pd
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from collections import defaultdict, Counter
from math import log
import ast


nltk.download('stopwords')
nltk.download('wordnet')


dataset_path = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'


df = pd.read_csv(dataset_path)

output_dir = '/Users/akshay/Desktop/IR/Assinment2/output'

# Define image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()  # Set the model to evaluation mode

def extract_image_features(urls):
    features_list = []
    try:
        # Convert the string representation of a list into an actual list
        urls = ast.literal_eval(urls) if isinstance(urls, str) and urls.startswith("[") else [urls]
        for url in urls:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img_t = image_transforms(img)
                img_t = img_t.unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    features = resnet(img_t)
                features_list.append(features.cpu().numpy().flatten())
            except requests.exceptions.RequestException as e:
                print(f"RequestException for URL {url}: {e}")
            except UnidentifiedImageError:
                print(f"UnidentifiedImageError: cannot identify image file from URL {url}.")
            except Exception as e:
                print(f"Unexpected error for URL {url}: {e}")
    except Exception as e:
        print(f"Error parsing URL string {urls}: {e}")
    return features_list



image_features = []
for index, row in df.iterrows():

    features_list = extract_image_features(row['Image'])
    image_features.extend(features_list)  # Use extend to flatten the list of lists


# Save the results
with open(os.path.join(output_dir, 'image_features.pkl'), 'wb') as f:
    pickle.dump(image_features, f)

print(f"Extracted and saved features for images.")