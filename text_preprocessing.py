import nltk
nltk.download('punkt_tab')


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ✅ Download NLTK stopwords and tokenizer
nltk.download("stopwords")
nltk.download("punkt")

# ✅ Load dataset
df = pd.read_csv("cleaned_meme_dataset.csv")

# ✅ Function to process text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words("english"))  # Load stopwords
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords & punctuation
    return " ".join(filtered_tokens)  # Join words back into a string

# ✅ Apply text preprocessing
df["Processed Captions"] = df["Meme Name"].astype(str).apply(preprocess_text)

# ✅ Save the updated dataset
df.to_csv("preprocessed_meme_dataset.csv", index=False)
print("✅ Text Preprocessing Complete! Dataset saved as 'preprocessed_meme_dataset.csv'.")
