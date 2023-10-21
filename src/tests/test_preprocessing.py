import pytest

from ..modules import preprocessing

def test_replace_abbreviations():
    # Basic Test Cases
    assert preprocessing.replace_abbreviations("i'm going to nyc") == "i am going to new york city"
    assert preprocessing.replace_abbreviations("lol, that was funny") == "laughing out loud, that was funny"
    assert preprocessing.replace_abbreviations("") == ""

    # Edge Test Cases
    assert preprocessing.replace_abbreviations("nyc and la") == "new york city and los angeles"
    assert preprocessing.replace_abbreviations("nyc, la, and sf") == "new york city, los angeles, and san francisco"
    assert preprocessing.replace_abbreviations("@nyc!") == "@new york city!"
    assert preprocessing.replace_abbreviations("nyc nyc nyc") == "new york city new york city new york city"

    # Error Handling Test Cases
    try:
        preprocessing.replace_abbreviations(None)
    except Exception as e:
        print(str(e))
        assert str(e) == "Error in replace_abbreviations: expected string or bytes-like object\nText: None"

    assert preprocessing.replace_abbreviations(None, ignore_errors=True) == None
    
def test_replace_multiple_spaces():
    # Basic Test Cases
    assert preprocessing.replace_multiple_spaces("This is  a  test") == "This is a test"
    assert preprocessing.replace_multiple_spaces("  This is a test  ") == "This is a test"
    assert preprocessing.replace_multiple_spaces("This is a test") == "This is a test"
    assert preprocessing.replace_multiple_spaces("") == ""

    # Edge Test Cases
    assert preprocessing.replace_multiple_spaces("This is \n a \n test") == "This is a test"
    assert preprocessing.replace_multiple_spaces("     ") == ""
    assert preprocessing.replace_multiple_spaces("\tThis is a test") == "This is a test"
    assert preprocessing.replace_multiple_spaces("This  is a  test! ") == "This is a test!"

    # Error Handling Test Cases
    try:
        preprocessing.replace_multiple_spaces(None)
    except Exception as e:
        assert str(e) == "Error in replace_multiple_spaces: expected string or bytes-like object\nText: None"

    assert preprocessing.replace_multiple_spaces(None, ignore_errors=True) == None


def test_handle_emojis():
    # Basic Test Cases
    assert preprocessing.handle_emojis("I love you 😍") == "I love you  smiling_face_with_heart-eyes "
    assert preprocessing.handle_emojis("This is funny 😂") == "This is funny  face_with_tears_of_joy "
    assert preprocessing.handle_emojis("") == ""

    # Edge Test Cases
    assert preprocessing.handle_emojis("This is 😍😂") == "This is  smiling_face_with_heart-eyes  face_with_tears_of_joy "
    assert preprocessing.handle_emojis("😍😂😍") == " smiling_face_with_heart-eyes  face_with_tears_of_joy  smiling_face_with_heart-eyes "
    assert preprocessing.handle_emojis("😍 This is 😂 a test 😍") == " smiling_face_with_heart-eyes  This is  face_with_tears_of_joy  a test  smiling_face_with_heart-eyes "

    # Error Handling Test Cases
    try:
        preprocessing.handle_emojis(None)
    except Exception as e:
        assert str(e) == "Error in handle_emojis: object of type 'NoneType' has no len()\nText: None"

    assert preprocessing.handle_emojis(None, ignore_errors=True) == None


def test_remove_urls():
    # Basic Test Cases
    assert preprocessing.remove_urls("Check out this website: http://example.com") == "Check out this website:  URL "
    assert preprocessing.remove_urls("Visit www.example.com for more info") == "Visit  URL  for more info"
    assert preprocessing.remove_urls("") == ""

    # Edge Test Cases
    assert preprocessing.remove_urls("This is a test URL: http://example.com/test?param=value") == "This is a test URL:  URL "
    assert preprocessing.remove_urls("URL with special characters: www.example.com/test?param=value#anchor") == "URL with special characters:  URL "
    assert preprocessing.remove_urls("Multiple URLs: http://example1.com and http://example2.com") == "Multiple URLs:  URL  and  URL "

    # Error Handling Test Cases
    try:
        preprocessing.remove_urls(None)
    except Exception as e:
        assert str(e) == "Error in remove_urls: expected string or bytes-like object\nText: None"

    assert preprocessing.remove_urls(None, ignore_errors=True) == None


def test_extract_url_info():
    # Basic Test Cases
    assert preprocessing.extract_url_info("Check out this website: http://example.com") == "Check out this website: example.com"
    assert preprocessing.extract_url_info("Visit www.example.com for more info") == "Visit example.com for more info"
    assert preprocessing.extract_url_info("") == ""

    # Edge Test Cases
    assert preprocessing.extract_url_info("This is a test URL: http://example.com/test?param=value") == "This is a test URL: example.com test param value"
    assert preprocessing.extract_url_info("URL with special characters: www.example.com/test?param=value#anchor") == "URL with special characters: example.com test param value anchor"
    assert preprocessing.extract_url_info("Multiple URLs: http://example1.com and http://example2.com") == "Multiple URLs: example1.com and example2.com"

    # Error Handling Test Cases
    try:
        preprocessing.extract_url_info(None)
    except Exception as e:
        assert str(e) == "Error in extract_url_info: expected string or bytes-like object\nText: None"

    assert preprocessing.extract_url_info(None, ignore_errors=True) == None


def test_remove_mentions():
    # Basic Test Cases
    assert preprocessing.remove_mentions("Hello @user!") == "Hello  MENTION !"
    assert preprocessing.remove_mentions("Hello world!") == "Hello world!"
    assert preprocessing.remove_mentions("") == ""

    # Edge Test Cases
    assert preprocessing.remove_mentions("Hello @user_123!") == "Hello  MENTION !"
    assert preprocessing.remove_mentions("Hello @user @user_123!") == "Hello  MENTION   MENTION !"
    assert preprocessing.remove_mentions("Hello @user! How are you?") == "Hello  MENTION ! How are you?"
    assert preprocessing.remove_mentions("Hello\n@user!") == "Hello\n MENTION !"

    # Error Handling Test Cases
    try:
        preprocessing.remove_mentions(None)
    except Exception as e:
        assert str(e) == "Error in remove_mentions: expected string or bytes-like object\nText: None"

    assert preprocessing.remove_mentions(None, ignore_errors=True) == None


def test_extract_mentions():
    # Basic Test Cases
    assert preprocessing.extract_mentions("Hello @user!") == "Hello  mention user !"
    assert preprocessing.extract_mentions("Hello world!") == "Hello world!"
    assert preprocessing.extract_mentions("") == ""

    # Edge Test Cases
    assert preprocessing.extract_mentions("Hello @user_123!") == "Hello  mention user_123 !"
    assert preprocessing.extract_mentions("Hello @user @user_123!") == "Hello  mention user   mention user_123 !"
    assert preprocessing.extract_mentions("Hello @user! How are you?") == "Hello  mention user ! How are you?"
    assert preprocessing.extract_mentions("Hello\n@user!") == "Hello\n mention user !"

    # Error Handling Test Cases
    try:
        preprocessing.extract_mentions(None)
    except Exception as e:
        assert str(e) == "Error in extract_mentions: expected string or bytes-like object\nText: None"

    assert preprocessing.extract_mentions(None, ignore_errors=True) == None


def test_remove_hashtags():
    # Basic Test Cases
    assert preprocessing.remove_hashtags("Check out this hashtag: #example") == "Check out this hashtag:  HASHTAG "
    assert preprocessing.remove_hashtags("No hashtags here!") == "No hashtags here!"
    assert preprocessing.remove_hashtags("") == ""

    # Edge Test Cases
    assert preprocessing.remove_hashtags("This is a test hashtag: #example#test") == "This is a test hashtag:  HASHTAG  HASHTAG "
    assert preprocessing.remove_hashtags("Hashtag with special characters: #example#test!") == "Hashtag with special characters:  HASHTAG  HASHTAG !"
    assert preprocessing.remove_hashtags("Multiple hashtags: #example1 and #example2") == "Multiple hashtags:  HASHTAG  and  HASHTAG "

    # Error Handling Test Cases
    try:
        preprocessing.remove_hashtags(None)
    except Exception as e:
        assert str(e) == "Error in remove_hashtags: expected string or bytes-like object\nText: None"

    assert preprocessing.remove_hashtags(None, ignore_errors=True) == None


def test_extract_hashtags():
    # Basic Test Cases
    assert preprocessing.extract_hashtags("Check out this hashtag: #example") == "Check out this hashtag:  hashtag example "
    assert preprocessing.extract_hashtags("No hashtags here!") == "No hashtags here!"
    assert preprocessing.extract_hashtags("") == ""

    # Edge Test Cases
    assert preprocessing.extract_hashtags("This is a test hashtag: #example#test") == "This is a test hashtag:  hashtag example  hashtag test "
    assert preprocessing.extract_hashtags("Hashtags with special characters: #example#test!") == "Hashtags with special characters:  hashtag example  hashtag test !"
    assert preprocessing.extract_hashtags("Multiple hashtags with newline: #example\n#test") == "Multiple hashtags with newline:  hashtag example \n hashtag test "

    # Error Handling Test Cases
    try:
        preprocessing.extract_hashtags(None)
    except Exception as e:
        assert str(e) == "Error in extract_hashtags: expected string or bytes-like object\nText: None"

    assert preprocessing.extract_hashtags(None, ignore_errors=True) == None


def test_remove_punctuations():
    # Basic Test Cases
    assert preprocessing.remove_punctuations("Check out this 11 punctuations: !@#$%^&*()") == "Check out this 11 punctuations            "
    assert preprocessing.remove_punctuations("No punctuations here!") == "No punctuations here "
    assert preprocessing.remove_punctuations("") == ""

    # Edge Test Cases
    assert preprocessing.remove_punctuations("Punctuation with special characters: !@#$%^&*(){}[]") == "Punctuation with special characters                "
    assert preprocessing.remove_punctuations("Punctuation with whitespace: !@# $%^ &*()") == "Punctuation with whitespace              "
    assert preprocessing.remove_punctuations("Punctuation with newline: !@#\n$%^&*()") == "Punctuation with newline     \n       "
    assert preprocessing.remove_punctuations("Numbers with dots and commas: 1,000.50") == "Numbers with dots and commas  1,000.50"
    assert preprocessing.remove_punctuations("Dot and comma not between digits: .,") == "Dot and comma not between digits    "

    # Error Handling Test Cases
    try:
        preprocessing.remove_punctuations(None)
    except Exception as e:
        assert str(e) == "Error in remove_punctuations: expected string or bytes-like object\nText: None"

    assert preprocessing.remove_punctuations(None, ignore_errors=True) == None


def test_text_preprocessing():
    # Basic Test Cases
    assert preprocessing.text_preprocessing("Check out this website") == "check website"
    assert preprocessing.text_preprocessing("Check out this website: http://example.com") == "check website example com"
    assert preprocessing.text_preprocessing("") == ""

    # Edge Test Cases
    assert preprocessing.text_preprocessing("This is a test URL: http://example.com/test?param=value") == "test url example com test param value"
    assert preprocessing.text_preprocessing("Hello @user!") == "hello mention user"
    assert preprocessing.text_preprocessing("Check out this hashtag: #example") == "check hashtag hashtag example"
    assert preprocessing.text_preprocessing("I love this emoji 😊") == "love emoji smiling_face_with_smiling_eyes"

    # Normalization Test Cases
    assert preprocessing.text_preprocessing("Running and run are not the same", lem=True, stem=False) == "running run"
    assert preprocessing.text_preprocessing("Running and run are the same", lem=False, stem=True) == "run run"
    assert preprocessing.text_preprocessing("Running and run are the same", lem=True, stem=True) == "run run"

    # Error Handling Test Cases
    try:
        preprocessing.text_preprocessing(None)
    except Exception as e:
        assert str(e) == "Error in text_processing: 'NoneType' object has no attribute 'lower'\nText: None"

    assert preprocessing.text_preprocessing(None, ignore_errors=True) == None
