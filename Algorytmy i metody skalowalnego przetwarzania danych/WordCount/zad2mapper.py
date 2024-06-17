import sys


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
for line in sys.stdin:
    cleaned_line = line.translate(str.maketrans('', '', string.punctuation + string.digits)).lower()
    words = [lemmatizer.lemmatize(word) for word in cleaned_line.split() if word not in stop_words]
    for word in words:
        print(f"{word}\t1")
