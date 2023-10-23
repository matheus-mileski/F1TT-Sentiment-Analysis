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
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "la" : "los angeles",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sf" : "san francisco",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "yolo" : "you only live once",
    "zzz" : "sleeping bored and tired",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


def replace_abbreviations(text, ignore_errors=False):
    """
    Replace abbreviations and contractions in a text.

    Parameters:
        text (str): Input text containing possible abbreviations.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.

    Returns:
        str: Text with abbreviations replaced.
    """
    if text == "":
        return text
    
    try:
        def replacer(match):
            return abbreviations.get(match.group(0).lower(), match.group(0))

        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b', re.IGNORECASE)
        return pattern.sub(replacer, text)

    except Exception as e:
        error = f"Error in replace_abbreviations: {str(e)}\nText: {text}"
    
    if ignore_errors:
        print(error)
        return text
    
    raise Exception(error)


def replace_multiple_spaces(text, ignore_errors=False):
    """
    Replace multiple spaces with a single space and trim leading/trailing spaces.

    Parameters:
        text (str): Input text that may contain multiple spaces.
        ignore_errors (bool, optional): Whether to ignore errors and 
                        return the original text if an error occurs. 
                        Defaults to False.

    Returns:
        str: Text with multiple spaces replaced by a single space.
    """
    if text == "":
        return text
    
    try:
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Strip whitespaces
        text = text.strip()
        return text

    except Exception as e:
        error = f"Error in replace_multiple_spaces: {str(e)}\nText: {text}"
    
    if ignore_errors:
        print(error)
        return text
    
    raise Exception(error)

    
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
        
        raise Exception(error)
    
    
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
        
        raise Exception(error)


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
            url_original = url
            if not url.startswith("http"):
                url = "http://"+url
                
            url = url.replace(r"]", "")
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace("www.", "")
            path = parsed_url.path.strip("/").replace("/", " ")
            params = parsed_url.query.replace("&", " ").replace("=", " ")
            anchor = parsed_url.fragment
            url_info = f"{domain} {path} {params} {anchor}".strip()
            text = text.replace(url_original, url_info)
        return text
    except Exception as e:
        if str(e) == "Invalid IPv6 URL":
            text = remove_urls(text, ignore_errors = False)
            return text
        
        error = f"Error in extract_url_info: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise Exception(error)
    
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
        return re.sub(r'@[A-Za-z0-9_]+', ' MENTION ', text)
    except Exception as e:
        error = f"Error in remove_mentions: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise Exception(error)


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
        def replace_mention(match):
            username = match.group(0)[1:]
            return f" mention {username} "
        
        mentions = re.findall(r'@[A-Za-z0-9_]+', text)
        for mention in mentions:
            username = mention[1:]
            text = re.sub(r'@\w+', replace_mention, text)
        return text
    except Exception as e:
        error = f"Error in extract_mentions: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise Exception(error)

        
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
        return re.sub(r'#\w+', ' HASHTAG ', text)
    except Exception as e:
        error = f"Error in remove_hashtags: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise Exception(error)
    


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
        hashtags = re.findall(r'#\w+', text)
        for hashtag in hashtags:
            topic = hashtag[1:]
            text = text.replace(hashtag, f" hashtag {topic} ")
        return text
    except Exception as e:
        error = f"Error in extract_hashtags: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise Exception(error)
        
def remove_punctuations(text, ignore_errors = False):
    """
    Remove punctuations from the text, while keeping dots and commas between numbers.
    
    Parameters:
        text (str): Input text which may contain punctuations.
        ignore_errors (bool): If True, suppresses errors and prints them instead. Default is False.
        
    Returns:
        str: Text with punctuations removed.
    """
    
    try:
        # Replace all non-word characters except dots, commas, and digits
        text = re.sub(r'[^\w\s.,\d+]', ' ', text)
        # Replace dots that are not between digits
        text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)
        # Replace commas that are not between digits
        text = re.sub(r'(?<!\d),|,(?!\d)', ' ', text)
        return text
    except Exception as e:
        error = f"Error in remove_punctuations: {str(e)}\nText: {text}"
        if ignore_errors:
            print(error)
            return text
        
        raise Exception(error)
    
def text_preprocessing(text, lem = True, stem = False, ignore_errors = False):
    """
    Preprocess the input text by applying various text cleaning, 
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
        text = text.lower()
        text = replace_abbreviations(text, ignore_errors)
        text = handle_emojis(text, ignore_errors)
        text = extract_url_info(text, ignore_errors)
        text = extract_mentions(text, ignore_errors)
        text = extract_hashtags(text, ignore_errors)
        text = remove_punctuations(text, ignore_errors)
        text = replace_multiple_spaces(text, ignore_errors)

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
        
        raise Exception(error)
