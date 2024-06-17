#!/usr/lib/python3.8/python
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


def mapper(input_data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    result = []
    for line in input_data:
        cleaned_line = line.translate(str.maketrans('', '', string.punctuation + string.digits)).lower()
        words = [lemmatizer.lemmatize(word) for word in cleaned_line.split() if word not in stop_words]
        for word in words:
            result.append(f"{word}\t1")
    return result
