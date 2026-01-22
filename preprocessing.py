import pandas as pd
import numpy as np
import unidecode
import re
from spacy.cli import download
import spacy
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import contractions
import os

# Download required models
try:
    spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextCleaner:
    """Clean and preprocess text data."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps."""
        self._validate_dataframe(df)
        df = df.copy()
        print('Starting text cleaning...')
        df['sms'] = (df['sms']
                     .str.lower() # Lowercase
                     .apply(unidecode.unidecode) # Remove accents
                     .str.replace(r'[^a-zA-Z\s]', '', regex=True) # Remove special characters 
                     .apply(lambda x: str(TextBlob(x).correct())) # Spelling correction
                     .apply(self._remove_repetition) # Remove character repetition
                     .apply(self._lemmatize) # Lemmatization
                    .apply(lambda x: contractions.fix(x)) # Expand contractions
                    .apply(lambda x: ' '.join([char for char in x.split() if char not in self.stop_words])) # Remove stop words
                    .loc[df.sms.map(len) > 0]) # Remove empty messages
        return df
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        required_cols = {'sms', 'is_spam'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    @staticmethod
    def _remove_repetition(text: str) -> str:
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def _lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return ' '.join(token.lemma_ for token in doc)
    
    def export_to_csv(self, df: pd.DataFrame, output_dir: str = "dataset/cleaned/") -> None:
        """Export cleaned dataframe to CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cleaned_spam.csv")
        df.to_csv(output_path, index=False)
        print(f"Data exported to {output_path}")
