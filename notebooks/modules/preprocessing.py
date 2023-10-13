import re
import emoji
import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse


# Instantiate objects outside of functions to avoid repeated instantiation
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def handle_emojis(text, ignore_errors = False):
    """
    Convert emojis into word representations.
    
    Parameters:
        text (str): Input text which may contain emojis.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: Text with emojis replaced by words.
    """
    
    try:
        return emoji.demojize(text, delimiters=(" ", " "))
    except Exception as e:
        error = f"Error in handle_emojis: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error
    
    
def remove_urls(text, ignore_errors = False):
    """
    Replace URLs in the text with a placeholder.

    Parameters:
        text (str): Input text which may contain URLs.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.

    Returns:
        str: Text with URLs replaced by a placeholder.
    """
    
    try:
        return re.sub(r'http\S+|www.\S+', ' URL ', text)
    except Exception as e:
        error = f"Error in remove_urls: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error


def extract_url_info(text, ignore_errors = False):
    """
    Extract and replace URLs with domain, path, and parameter information.
    
    Parameters:
        text (str): Input text which may contain URLs.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: Text with URLs replaced by extracted information.
    """
    
    try:
        urls = re.findall(r'http\S+|www.\S+', text)
        for url in urls:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace("www.", "")
            path = parsed_url.path.strip("/").replace("/", " ")
            params = parsed_url.query.replace("&", " ").replace("=", " ")
            url_info = f" {domain} {path} {params} "
            text = text.replace(url, url_info)
        return text
    except Exception as e:
        error = f"Error in extract_url_info: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error
    
def remove_mentions(text, ignore_errors=False):
    """
    Replace mentions in the text with a placeholder.

    Parameters:
        text (str): Input text which may contain mentions.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.

    Returns:
        str: Text with mentions replaced by a placeholder.
    """
    
    try:
        return re.sub(r'@\S+', ' MENTION ', text)
    except Exception as e:
        error = f"Error in remove_mentions: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error


def extract_mentions(text, ignore_errors = False):
    """
    Extract and replace mentions (@username) with the word "mention" followed 
    by the username.
    
    Parameters:
        text (str): Input text which may contain mentions.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: Text with mentions replaced by extracted information.
    """
    
    try:
        mentions = re.findall(r'@\S+', text)
        for mention in mentions:
            username = mention[1:]
            text = text.replace(mention, f" mention {username} ")
        return text
    except Exception as e:
        error = f"Error in extract_mentions: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error

        
def remove_hashtags(text, ignore_errors = False):
    """
    Replace hashtags in the text with a placeholder.

    Parameters:
        text (str): Input text which may contain hashtags.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.

    Returns:
        str: Text with hashtags replaced by a placeholder.
    """
    
    try:
        return re.sub(r'#\S+', ' HASHTAG ', text)
    except Exception as e:
        error = f"Error in remove_hashtags: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error
    


def extract_hashtags(text, ignore_errors = False):
    """
    Extract and replace hashtags (#hashtag) with the word "hashtag" followed 
    by the topic.
    
    Parameters:
        text (str): Input text which may contain hashtags.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: Text with hashtags replaced by extracted information.
    """
    
    try:
        hashtags = re.findall(r'#\S+', text)
        for hashtag in hashtags:
            topic = hashtag[1:]
            text = text.replace(hashtag, f" hashtag {topic} ")
        return text
    except Exception as e:
        error = f"Error in extract_hashtags: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error
        
def remove_punctuations(text, ignore_errors = False):
    """
    Remove punctuations from the text.
    
    Parameters:
        text (str): Input text which may contain punctuations.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: Text with punctuations removed.
    """
    
    try:
        return re.sub(r'[^\w\s]', ' ', text)
    except Exception as e:
        error = f"Error in remove_punctuations: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error
    
def text_preprocessing(text, lem = True, stem = False, ignore_errors = False):
    """
    Preprocesses the input text by applying various text cleaning, 
    tokenization, and normalization steps.
    
    Parameters:
        text (str): The input text to be processed.
        lem (bool): If True, applies lemmatization. Default is True.
        stem (bool): If True, applies stemming. Default is False.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: The preprocessed text.
    """
    
    try:
        text = handle_emojis(text, ignore_errors)
        text = extract_url_info(text, ignore_errors)
        text = extract_mentions(text, ignore_errors)
        text = extract_hashtags(text, ignore_errors)
        text = remove_punctuations(text, ignore_errors)
        text = text.lower()

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]

        if lem:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        if stem:
            tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)
    
    except Exception as e:
        error = f"Error in text_processing: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise error