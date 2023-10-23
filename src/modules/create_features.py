import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from textblob import TextBlob
from scipy.sparse import csr_matrix, hstack


def generate_tfidf_features(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix


def generate_word_embeddings(texts):
    # Load the Word2Vec model
    word2vec_model_path = '../word2vec_data/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

    # Create a matrix of word embeddings
    embeddings_matrix = []
    for text in texts:
        words = text.split()
        embeddings = [model[word] for word in words if word in model]
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
        else:
            mean_embedding = np.zeros(300)
        embeddings_matrix.append(mean_embedding)

    return csr_matrix(embeddings_matrix)

def generate_dl_word_embeddings(texts):
    word2vec_model_path = '../word2vec_data/GoogleNews-vectors-negative300.bin'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    
    def get_avg_word2vec_vector(text, model, vector_size):
        words = text.split()
        word_vectors = [model[word] for word in words if word in model]
        if len(word_vectors) == 0:
            return np.zeros(vector_size)
        else:
            return np.mean(word_vectors, axis=0)

    vector_size = 300  # Size of the Word2Vec vectors in the Google News model
    word2vec_features = np.array([get_avg_word2vec_vector(text, word2vec_model, vector_size) for text in texts])
    
    return word2vec_features


def generate_sentiment_scores(texts):
    sentiment_scores = [TextBlob(text).sentiment.polarity for text in texts]
    return csr_matrix(np.array(sentiment_scores).reshape(-1, 1))

def generate_dl_sentiment_scores(texts):
    def get_sentiment_score(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    sentiment_scores = np.array([get_sentiment_score(text) for text in texts])
    return sentiment_scores

def save_features(feature_matrix, labels, feature_matrix_path, labels_path):
    joblib.dump(feature_matrix, feature_matrix_path)
    
    if labels_path is not None:
        joblib.dump(labels, labels_path)
    

def create_features(df, feature_matrix_path, labels_path, test):
    if test:
        if 'polarity' in df.columns:
            df_0 = df.filter(df['polarity'] == 0).limit(800)
            df_1 = df.filter(df['polarity'] == 1).limit(800)
            df = pl.concat([df_0, df_1])
        else:
            df = df.limit(1600)

    print("Generate TF-IDF features")
    tfidf_matrix = generate_tfidf_features(df['preprocessed_text'].to_list())

    print("Generate word embeddings")
    embeddings_matrix = generate_word_embeddings(df['preprocessed_text'].to_list())

    print("Generate sentiment scores")
    sentiment_scores = generate_sentiment_scores(df['preprocessed_text'].to_list())

    print("Combine all features into a single matrix")
    feature_matrix = hstack([tfidf_matrix, embeddings_matrix, sentiment_scores])

    print("Save the feature matrix and labels to a file")
    if labels_path is not None:
        save_features(feature_matrix, df['polarity'].to_numpy(), feature_matrix_path, labels_path)
    else:
        save_features(feature_matrix, None, feature_matrix_path, None)

        
