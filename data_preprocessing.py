import pandas as pd
import os
import re  # ✅ Import re for regular expressions
import requests
from tqdm import tqdm

import pandas as pd

import pandas as pd

df = pd.read_csv("cleaned_meme_dataset.csv")
print(df.head())  # Show first 5 rows


df = pd.read_csv("preprocessed_meme_dataset.csv")
print("Columns in dataset:", df.columns)

# Check if 'Processed Captions' column exists
if "Processed Captions" not in df.columns:
    print("❌ 'Processed Captions' column is missing!")
    exit()

# Display first few rows
print(df.head())


# Load the dataset
df = pd.read_csv("meme_dataset.csv")  # Ensure this file contains Meme Name, Image URL

# ✅ Create the memes directory if it doesn’t exist
os.makedirs("memes", exist_ok=True)

# Download images
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_url = row["Image URL"]
    img_name = re.sub(r'[<>:"/\\|?*]', "", row["Meme Name"]).replace(" ", "_") + ".jpg"  # ✅ Fix
    img_path = os.path.join("memes", img_name)
    
    if not os.path.exists(img_path):  # Avoid redownloading
        try:
            response = requests.get(img_url, stream=True, timeout=5)
            with open(img_path, "wb") as img_file:
                img_file.write(response.content)
            df.at[index, "Local Image Path"] = img_path
        except Exception as e:
            print(f"⚠️ Error downloading {img_url}: {e}")

# Save the updated dataset
df.to_csv("cleaned_meme_dataset.csv", index=False)
print("✅ Dataset cleaned & images downloaded.")



