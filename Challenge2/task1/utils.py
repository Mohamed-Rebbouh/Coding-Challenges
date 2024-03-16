import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
def remove_urls(text, replacement_text=""):
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub(replacement_text, text)


def remove_twitter_handles(text, replacement_text=""):
    pattern = re.compile(r"@[\w]+")
    return pattern.sub(replacement_text, text)


def remove_twitter_rt(text, replacement_text=""):
    pattern = re.compile(r"^RT|\s+RT\s+")
    return pattern.sub(replacement_text, text)


def remove_alphanumerics(text, replacement_text=" "):
    pattern = re.compile(r"[^A-Za-z0-9']+")
    return pattern.sub(replacement_text, text)


def remove_multiple_whitespaces(text, replacement_text=" "):
    pattern = re.compile(r"\s{2,}")
    return pattern.sub(replacement_text, text)


def decode_html_character_references(text):
    import html
    return html.unescape(text)

def tokenize(doc):
    return word_tokenize(doc)

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

def remove_stopwords(doc):
    stops = set(stopwords.words("english"))
    stops.add("rt")
    return [token for token in doc if token not in stops]


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"ain\'t", "are not", phrase)
    phrase = re.sub(r"shan\'t", "shall not", phrase)
    phrase = re.sub(r"ma\'am", "maam", phrase)
    phrase = re.sub(r"y\'all", "you all", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# with open('count_vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)


# def clean(text):
#     nltk.download('wordnet')
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     lemmatizer = WordNetLemmatizer()
#     txt=remove_urls(text)
#     txt=remove_twitter_handles(txt)
#     txt=remove_twitter_rt(txt)
#     txt=remove_alphanumerics(txt)
#     txt=remove_multiple_whitespaces(txt)
#     txt=decode_html_character_references(txt)
#     txt=txt.lower()
#     txt=[decontracted(word) for word in txt.split(" ")]
#     txt=[lemmatizer.lemmatize(word) for word in txt]
#     txt=" ".join(txt)
#     txt=word_tokenize(txt)
#     txt=remove_stopwords(txt)
#     txt=" ".join(txt)
#     return txt

def clean(text):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    lemmatizer = WordNetLemmatizer()
    text = remove_urls(text)
    text = remove_twitter_handles(text)
    text = remove_twitter_rt(text)
    text = remove_alphanumerics(text)
    text = remove_multiple_whitespaces(text)
    text = decode_html_character_references(text)
    text = text.lower()
    text = decontracted(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    text = " ".join(tokens)
    return text


def import_vec_(txt):
    txt=clean(txt)
    with open('count_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer.transform([txt])

def import_model(txt):
    txt=import_vec_(txt)
    with open('random_forest_model.pkl', 'rb') as f:
        model=pickle.load(f)
    return model.predict(txt)