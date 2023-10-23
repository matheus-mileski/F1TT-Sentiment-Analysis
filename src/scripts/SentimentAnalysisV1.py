import os
import sys
import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from time import time
from pathlib import Path
from src.modules.preprocessing import text_preprocessing
from src.modules.create_features import create_features, generate_dl_sentiment_scores, generate_dl_word_embeddings
from src.modules.generate_plot import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, plot_val_loss_accuracy
from pydrive.auth import GoogleAuth 
from pydrive.drive import GoogleDrive
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping


def preprocess_s140_dataset(raw_data_path, preprocessed_data_path):
    # Load the dataset
    encode = 'ISO-8859-1'
    df = pl.read_csv(raw_data_path, has_header=False, encoding=encode, new_columns=["polarity", "id", "date", "flag", "user", "text"])
    
    # Create a new column for the preprocessed text
    df = df.with_columns(
        df["text"].map_elements(lambda text: text_preprocessing(text)).alias("preprocessed_text"),
        df["polarity"].apply(lambda x: 1 if x == 4 else x).alias("polarity")
    )

    # Save the preprocessed dataset
    df.write_csv(preprocessed_data_path)


def preprocess_f1_dataset(raw_data_path, preprocessed_f1_data_path):
    # Load the dataset
    dtypes = {
        'user_name':pl.Utf8,
        'user_location':pl.Utf8,
        'user_description':pl.Utf8,
        'user_created':pl.Utf8,
        'user_followers':pl.Utf8,
        'user_friends':pl.Utf8,
        'user_favourites':pl.Utf8,
        'user_verified':pl.Utf8,
        'date':pl.Utf8,
        'text':pl.Utf8,
        'hashtags':pl.Utf8,
        'source':pl.Utf8,
        'is_retweet':pl.Utf8
    }
    encode = 'Utf8'
    df = pl.read_csv(raw_data_path, dtypes=dtypes, encoding=encode)
    
    # Create a new column for the preprocessed text
    df = df.with_columns(
        df["text"].map_elements(lambda text: text_preprocessing(text)).alias("preprocessed_text")
    )

    # Save the preprocessed dataset
    df.write_csv(preprocessed_f1_data_path)

def create_s140_features(preprocessed_data_path_s140, feature_matrix_path_s140, labels_path_s140, test):
    # Load the dataset
    encode = 'ISO-8859-1'
    df = pl.read_csv(preprocessed_data_path_s140, encoding=encode)
    
    create_features(df, feature_matrix_path_s140, labels_path_s140, test)

def create_f1_features(preprocessed_data_path_f1, feature_matrix_path_f1, labels_path_f1, test):
    # Load the dataset
    dtypes = {
        'user_name':pl.Utf8,
        'user_location':pl.Utf8,
        'user_description':pl.Utf8,
        'user_created':pl.Utf8,
        'user_followers':pl.Utf8,
        'user_friends':pl.Utf8,
        'user_favourites':pl.Utf8,
        'user_verified':pl.Utf8,
        'date':pl.Utf8,
        'text':pl.Utf8,
        'hashtags':pl.Utf8,
        'source':pl.Utf8,
        'is_retweet':pl.Utf8,
        'preprocessed_text':pl.Utf8
    }
    encode = 'Utf8'
    df = pl.read_csv(preprocessed_data_path_f1, dtypes=dtypes, encoding=encode)
    
    create_features(df, feature_matrix_path_f1, None, test)

def evaluate_pca(feature_matrix_path, labels_path, pca_components_path, pca_variance_ratio_path, pca_plot_path, pca_components_reduced_path, pca_variance_ratio_reduced_path, pca_plot_reduced_path):
    # Load the feature matrix and labels
    feature_matrix = joblib.load(feature_matrix_path)
    labels = joblib.load(labels_path)
    
    # Standardize the feature matrix
    scaler = StandardScaler(with_mean=False)
    feature_matrix_std = scaler.fit_transform(feature_matrix)
    
    # Apply TruncatedSVD (PCA for sparse matrices) with maximum components
    svd = TruncatedSVD(n_components=min(feature_matrix_std.shape) - 1)
    svd_components = svd.fit_transform(feature_matrix_std)
    
    # Determine the number of components required to retain 95% variance
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    # Plot the explained variance ratio
    plt.plot(cumulative_variance)
    plt.axvline(x=n_components_95, color='r', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.savefig(pca_plot_path)
    plt.show()
    
    # Save the PCA components and variance ratio for 95% variance
    joblib.dump(svd_components[:, :n_components_95], pca_components_path)
    joblib.dump(svd.explained_variance_ratio_[:n_components_95], pca_variance_ratio_path)
    
    # Apply TruncatedSVD (PCA for sparse matrices) for reduced components
    svd = TruncatedSVD(n_components=2)  # Reduce to 2D for visualization
    svd_components = svd.fit_transform(feature_matrix_std)
    
    # Plot the reduced feature space with different classes highlighted
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(svd_components[:, 0], svd_components[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Reduced Feature Space with Different Classes Highlighted')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.savefig(pca_plot_path)
    plt.show()
    
    # Save the PCA components and variance ratio
    joblib.dump(svd_components, pca_components_reduced_path)
    joblib.dump(svd.explained_variance_ratio_, pca_variance_ratio_reduced_path)

def evaluate_models(feature_matrix_path, labels_path, models_path, predictions_path, test_size=0.2):
    # Load the feature matrix and labels
    feature_matrix = joblib.load(feature_matrix_path)
    labels = joblib.load(labels_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=test_size, random_state=42)

    # Initialize models
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'MLPClassifier': MLPClassifier(),
    }

    # Initialize metrics
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1 Score': f1_score,
    }

    # Train and evaluate models
    for model_name, model in models.items():
        model_file = f"{models_path}/{model_name}_metrics.txt"
        model_pickle_file = f"{models_path}/{model_name}_model.pkl"
        if Path(model_file).exists() and Path(model_pickle_file).exists():
            print(f"{model_name} already evaluated. Skipping...")
            continue
        print(f"Training and evaluating {model_name}...")
        if model_name == 'Naive Bayes':
            model.fit(X_train.toarray(), y_train)
            y_pred = model.predict(X_test.toarray())
            y_pred_prob = model.predict_proba(X_test.toarray())[:, 1]

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

        joblib.dump(model, model_pickle_file)  # Save the trained model
        joblib.dump(y_pred, f"{predictions_path}/{model_name}_predictions.pkl")
        
        plot_confusion_matrix(y_test, y_pred, f"{predictions_path}/{model_name}_confusion_matrix.png")
        plot_roc_curve(y_test, y_pred_prob, f"{predictions_path}/{model_name}_roc_curve.png")
        plot_precision_recall_curve(y_test, y_pred_prob, f"{predictions_path}/{model_name}_precision_recall_curve.png")
        
        with open(model_file, 'w') as f:
            for metric_name, metric in metrics.items():
                score = metric(y_test, y_pred)
                f.write(f"{metric_name}: {score}\n")
            f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
            f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")


def create_lstm_model(max_words, max_len, vector_size):
    sequence_input = Input(shape=(max_len,), dtype='int32')
    embedding_layer = Embedding(max_words, 128, input_length=max_len)(sequence_input)
    lstm_layer = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedding_layer)
    
    word2vec_input = Input(shape=(vector_size,))
    dense_word2vec = Dense(64, activation='relu')(word2vec_input)
    
    sentiment_input = Input(shape=(1,))
    dense_sentiment = Dense(64, activation='relu')(sentiment_input)
    
    combined = concatenate([lstm_layer, dense_word2vec, dense_sentiment])
    
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[sequence_input, word2vec_input, sentiment_input], outputs=output)
    return model


def create_cnn_model(max_words, max_len, vector_size):
    sequence_input = Input(shape=(max_len,), dtype='int32')
    embedding_layer = Embedding(max_words, 128, input_length=max_len)(sequence_input)
    conv1d_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
    maxpooling_layer = MaxPooling1D(5)(conv1d_layer)
    flatten_layer = Flatten()(maxpooling_layer)
    
    word2vec_input = Input(shape=(vector_size,))
    dense_word2vec = Dense(64, activation='relu')(word2vec_input)
    
    sentiment_input = Input(shape=(1,))
    dense_sentiment = Dense(64, activation='relu')(sentiment_input)
    
    combined = concatenate([flatten_layer, dense_word2vec, dense_sentiment])
    
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[sequence_input, word2vec_input, sentiment_input], outputs=output)
    return model


def evaluate_deep_learning_models(preprocessed_data_path, models_path, predictions_path, test_size=0.2, max_words=10000, max_len=100, epochs=100, batch_size=32, validation_split=0.2, test=False):
    # Load the dataset
    encode = 'ISO-8859-1'
    df = pl.read_csv(preprocessed_data_path, encoding=encode)
    
    if test:
        if 'polarity' in df.columns:
            df_0 = df.filter(df['polarity'] == 0).limit(800)
            df_1 = df.filter(df['polarity'] == 1).limit(800)
            df = pl.concat([df_0, df_1])
        
    # Get the preprocessed text and labels
    texts = df['preprocessed_text'].to_list()
    labels = df['polarity'].to_list()

    # Tokenize the texts
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word2vec_features = generate_dl_word_embeddings(texts)
    sentiment_scores = generate_dl_sentiment_scores(texts)
    X_sequences = pad_sequences(sequences, maxlen=max_len)
    y = np.array(labels)

    # Split the dataset into training and testing sets
    X_train_sequences, X_test_sequences, y_train, y_test = train_test_split(X_sequences, y, test_size=test_size, random_state=42)
    X_train_word2vec, X_test_word2vec, _, _ = train_test_split(word2vec_features, y, test_size=test_size, random_state=42)
    X_train_sentiment, X_test_sentiment, _, _ = train_test_split(sentiment_scores, y, test_size=test_size, random_state=42)

    # Initialize deep learning models
    vector_size = word2vec_features.shape[1]
    models = {
        'LSTM': create_lstm_model(max_words, max_len, vector_size),
        'CNN': create_cnn_model(max_words, max_len, vector_size),
    }

    # Train and evaluate deep learning models
    for model_name, model in models.items():
        model_file = f"{models_path}/{model_name}_metrics.txt"
        model_keras_file = f"{models_path}/{model_name}_model.keras"
        if Path(model_file).exists() and Path(model_keras_file).exists():
            print(f"{model_name} already evaluated. Skipping...")
            continue
        print(f"Training and evaluating {model_name}...")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(
            [X_train_sequences, X_train_word2vec, X_train_sentiment], 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[early_stopping]
        )
        model.save(model_keras_file)  # Save the trained model
        y_pred = model.predict([X_test_sequences, X_test_word2vec, X_test_sentiment])
        y_pred = (y_pred > 0.5).astype(int).reshape(-1)
        joblib.dump(y_pred, f"{predictions_path}/{model_name}_predictions.pkl")
        
        y_pred_prob = model.predict([X_test_sequences, X_test_word2vec, X_test_sentiment]).ravel()
        
        plot_val_loss_accuracy(history, f"{predictions_path}/{model_name}_history.png")
        plot_confusion_matrix(y_test, y_pred, f"{predictions_path}/{model_name}_confusion_matrix.png")
        plot_roc_curve(y_test, y_pred_prob, f"{predictions_path}/{model_name}_roc_curve.png")
        plot_precision_recall_curve(y_test, y_pred_prob, f"{predictions_path}/{model_name}_precision_recall_curve.png")
        
        with open(model_file, 'w') as f:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
            f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")


def export_data_to_google_drive(google_drive_path):
    # Code for exporting data to Google Drive
    pass

def run_models_on_f1_dataset(preprocessed_f1_data_path, models_path, predictions_path):
    # Code for running models on the F1 dataset
    pass


def main():
    print("Start script")
    
    test = False
    if len(sys.argv[1:]):
        args = sys.argv[1:]
        if args[0] == '-test':
            print("Running test environment")
            test = True
    
    # Paths and filenames
    raw_data_path_s140 = "data/training.1600000.processed.noemoticon.csv"
    raw_data_path_f1 = "data/F1_tweets.csv"
    preprocessed_data_path_s140 = "data/preprocessed_training.1600000.processed.noemoticon.csv"
    preprocessed_data_path_f1 = "data/preprocessed_F1_tweets.csv"
    feature_matrix_path_s140 = "data/feature_matrix_s140.pkl"
    feature_matrix_path_f1 = "data/feature_matrix_f1.pkl"
    labels_path_s140 = "data/labels_s140.pkl"
    pca_components_path = "data/pca_components.pkl"
    pca_variance_ratio_path = "data/pca_variance_ratio.pkl"
    pca_plot_path = "data/pca_explained_variance_ratio.png"
    pca_components_reduced_path = "data/pca_components.pkl"
    pca_variance_ratio_reduced_path = "data/pca_variance_ratio.pkl"
    pca_plot_reduced_path = 'data/pca_reduced_feature_space.png'
    models_path = "data/models"
    predictions_path = "data/predictions"
    # google_drive_path = "path/to/google/drive/folder"

    # 1. Preprocessing
    if not Path(preprocessed_data_path_s140).exists():
        print("Preprocessing Sentiment140 dataset")
        start = time()
        preprocess_s140_dataset(raw_data_path_s140, preprocessed_data_path_s140)
        end = time()
        print(f"Sentiment140 preprocessed.\nTime elapsed: {(end - start)/60} min")

    if not Path(preprocessed_data_path_f1).exists():
        print("Preprocessing F1TT dataset")
        start = time()
        preprocess_f1_dataset(raw_data_path_f1, preprocessed_data_path_f1)
        end = time()
        print(f"F1TT preprocessed.\nTime elapsed: {(end - start)/60} min")

    # 2. Feature Creation
    if not Path(feature_matrix_path_s140).exists() or not Path(labels_path_s140).exists():
        print("Create NLP Features for Sentiment140 dataset")
        start = time()
        create_s140_features(preprocessed_data_path_s140, feature_matrix_path_s140, labels_path_s140, test)
        end = time()
        print(f"NLP Features for Sentiment140 created.\nTime elapsed: {(end - start)/60} min")
        
    if not Path(feature_matrix_path_f1).exists():
        print("Create NLP Features for F1 dataset")
        start = time()
        create_f1_features(preprocessed_data_path_f1, feature_matrix_path_f1, None, test)
        end = time()
        print(f"NLP Features for F1TT created.\nTime elapsed: {(end - start)/60} min")
        
    # # 3. PCA Evaluation
    # if not Path(pca_components_path).exists() or not Path(pca_variance_ratio_path).exists():
    #     print("Evaluate PCA for Sentiment140 dataset")
    #     start = time()
    #     evaluate_pca(feature_matrix_path_s140, labels_path_s140, pca_components_path, pca_variance_ratio_path, pca_plot_path, pca_components_reduced_path, pca_variance_ratio_reduced_path, pca_plot_reduced_path)
    #     end = time()
    #     print(f"PCA for Sentiment140 evaluated.\nTime elapsed: {(end - start)/60} min")

    # 4. Model Evaluation
    if not Path(models_path).exists():
        os.mkdir(models_path)
        
    if not Path(predictions_path).exists():
        os.mkdir(predictions_path)
        
    print("Evaluating machine learning models...")
    start = time()
    evaluate_models(feature_matrix_path_s140, labels_path_s140, models_path, predictions_path)
    end = time()
    print(f"Machine learning models evaluated.\nTime elapsed: {(end - start)/60} min")

    print("Evaluating deep learning models...")
    start = time()
    evaluate_deep_learning_models(preprocessed_data_path_s140, models_path, predictions_path, test=test)
    end = time()
    print(f"Deep learning models evaluated.\nTime elapsed: {(end - start)/60} min")


    # # 5. Data Export
    # export_data_to_google_drive(google_drive_path)

    # # 6. Run Models on F1 Dataset
    # run_models_on_f1_dataset(preprocessed_f1_data_path, models_path, predictions_path)

if __name__ == "__main__":
    main()