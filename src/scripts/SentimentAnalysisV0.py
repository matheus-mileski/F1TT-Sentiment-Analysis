print("Importing dependencies")
import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
        ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
        ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
        ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
        '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
        '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
        ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


def model_Evaluate(model, fpr, tpr, y_test, y_pred):
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg ','False Pos ', 'False Neg ','True Pos ']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    # labels = [f'{v1} {v2}' for v1, v2 in zip(group_names,group_percentages)]
    # labels = np.asarray(labels).reshape(2,2)
    # sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    # xticklabels = categories, yticklabels = categories)
    # plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    # plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    # plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC CURVE')
    # plt.legend(loc="lower right")
    # plt.show()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Return metrics as a dictionary
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': roc_auc,
        'True Neg ': group_percentages[0],
        'False Pos ': group_percentages[1],
        'False Neg ': group_percentages[2],
        'True Pos ': group_percentages[3]
    }

    return metrics_dict

def text_preprocessing(text):
    
    """ converting text to lower case """
    text = text.lower()
    
    for emoji in emojis.keys():
        text = text.replace(emoji, "emoji:" + emojis[emoji])
    
    """ replace the urls to URL """
    text = re.sub("http:\S+|www.\S", "URL", text)
    
    """ removing the punctuations """
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    """ stopwords removal """
    STOPWORDS = set(stopwords.words('english'))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

    """ tokenization """
    tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
    text = " ".join(tokenizer.tokenize(text))
    
    """ lemmatization """
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in str(text).split()])
    
    """ stemming """
    st = PorterStemmer()
    text = " ".join([st.stem(word) for word in str(text).split()])
    
    return text
    
if __name__ == "__main__":
    print("Running main")
    
    print("Read the CSV file")
    columns  = ["polarity", "id", "date", "flag", "user", "text"]
    encode = "ISO-8859-1"
    s140 = pd.read_csv("../data/training.1600000.processed.noemoticon.csv",  encoding=encode , names=columns)
    s140.drop(columns=['id', 'flag', 'user'], inplace=True)

    # display(s140.head())

    print("Convert the date column to datetime and UTC timezone")
    s140['date'] = s140['date'].str.replace(" PDT", "")
    s140['date'] = pd.to_datetime(s140['date'], format="%a %b %d %H:%M:%S %Y")
    s140['date'] = s140['date'] + pd.Timedelta(hours=7)
    s140['date'] = s140['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
    s140['date'] = pd.to_datetime(s140['date'])

    s140['polarity'] = s140['polarity'].replace(4,1)

    # Print the dataframe and its infos
    # display(s140.info(verbose=True, show_counts=True))
    # s140.head()


    # s140.polarity.value_counts()

    data_pos = s140[s140['polarity'] == 1]
    data_neg = s140[s140['polarity'] == 0]

    #data_pos = data_pos.iloc[:int(100000)]
    #data_neg = data_neg.iloc[:int(100000)]

    df = pd.concat([data_pos, data_neg])

    print("Data cleaning")
    # Remover:
    # - stopwords
    # - emojis (o dataset de treino não possui, mas o de validação sim)
    # - urls
    # - pontuação

    df["original_length"] = df["text"].apply(len)
    df["text_processed"] = df["text"].apply(lambda text : text_preprocessing(text))
    print("Exporting processed tweets.")
    df.to_csv("processed_tweets.csv")
    #df.head()


    # data_neg = df.query('polarity==0')['text_processed']
    # plt.figure(figsize = (20,20))
    # wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
    #             collocations=False).generate(" ".join(data_neg))
    # plt.imshow(wc)


    # data_pos = df.query('polarity==1')['text_processed']
    # plt.figure(figsize = (20,20))
    # wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
    #             collocations=False).generate(" ".join(data_pos))
    # plt.imshow(wc)


    X=df.text_processed
    y=df.polarity

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 49)

    print("TF-IDF")
    vectorizer_1_3_gram = TfidfVectorizer(ngram_range=(1,3))
    vectorizer_1_3_gram.fit(X_train)


    vectorizer_1_2_gram = TfidfVectorizer(ngram_range=(1,2))
    vectorizer_1_2_gram.fit(X_train)


    vectorizer_2_3_gram = TfidfVectorizer(ngram_range=(2,3))
    vectorizer_2_3_gram.fit(X_train)


    vectorizer_1_gram = TfidfVectorizer(ngram_range=(1,1))
    vectorizer_1_gram.fit(X_train)


    vectorizer_2_gram = TfidfVectorizer(ngram_range=(2,2))
    vectorizer_2_gram.fit(X_train)


    vectorizer_3_gram = TfidfVectorizer(ngram_range=(3,3))
    vectorizer_3_gram.fit(X_train)


    X_train_1_3_gram = vectorizer_1_3_gram.transform(X_train)
    X_test_1_3_gram  = vectorizer_1_3_gram.transform(X_test)


    X_train_1_2_gram = vectorizer_1_2_gram.transform(X_train)
    X_test_1_2_gram  = vectorizer_1_2_gram.transform(X_test)


    X_train_2_3_gram = vectorizer_2_3_gram.transform(X_train)
    X_test_2_3_gram  = vectorizer_2_3_gram.transform(X_test)


    X_train_1_gram = vectorizer_1_gram.transform(X_train)
    X_test_1_gram  = vectorizer_1_gram.transform(X_test)


    X_train_2_gram = vectorizer_2_gram.transform(X_train)
    X_test_2_gram  = vectorizer_2_gram.transform(X_test)


    X_train_3_gram = vectorizer_3_gram.transform(X_train)
    X_test_3_gram  = vectorizer_3_gram.transform(X_test)

    print("Start training")
    results_df = pd.DataFrame(columns=['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'True Neg ','False Pos ', 'False Neg ','True Pos '])

    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train_1_3_gram, y_train)
    y_pred1 = BNBmodel.predict(X_test_1_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    metrics_dict = model_Evaluate(BNBmodel, fpr, tpr, y_test, y_pred1)

    results_df = results_df.append({'Model': 'BernoulliNB', 'Dataset': 'X_train_1_3_gram', **metrics_dict}, ignore_index=True)



    SVCmodel = LinearSVC(dual='auto')
    SVCmodel.fit(X_train_1_3_gram, y_train)
    y_pred2 = SVCmodel.predict(X_test_1_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    metrics_dict = model_Evaluate(SVCmodel, fpr, tpr, y_test, y_pred2)

    results_df = results_df.append({'Model': 'LinearSVC', 'Dataset': 'X_train_1_3_gram', **metrics_dict}, ignore_index=True)



    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train_1_3_gram, y_train)
    y_pred3 = LRmodel.predict(X_test_1_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    metrics_dict = model_Evaluate(LRmodel, fpr, tpr, y_test, y_pred3)

    results_df = results_df.append({'Model': 'LogisticRegression', 'Dataset': 'X_train_1_3_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=49,
        n_jobs=-1
    )
    RFmodel.fit(X_train_1_3_gram, y_train)
    y_pred4 = RFmodel.predict(X_test_1_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred4)

    results_df = results_df.append({'Model': 'RandomForestClassifier1', 'Dataset': 'X_train_1_3_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_jobs=-1
    )
    RFmodel.fit(X_train_1_3_gram, y_train)
    y_pred5 = RFmodel.predict(X_test_1_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred5)

    results_df = results_df.append({'Model': 'RandomForestClassifier2', 'Dataset': 'X_train_1_3_gram', **metrics_dict}, ignore_index=True)



    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train_1_2_gram, y_train)
    y_pred1 = BNBmodel.predict(X_test_1_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    metrics_dict = model_Evaluate(BNBmodel, fpr, tpr, y_test, y_pred1)

    results_df = results_df.append({'Model': 'BernoulliNB', 'Dataset': 'X_train_1_2_gram', **metrics_dict}, ignore_index=True)



    SVCmodel = LinearSVC(dual='auto')
    SVCmodel.fit(X_train_1_2_gram, y_train)
    y_pred2 = SVCmodel.predict(X_test_1_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    metrics_dict = model_Evaluate(SVCmodel, fpr, tpr, y_test, y_pred2)

    results_df = results_df.append({'Model': 'LinearSVC', 'Dataset': 'X_train_1_2_gram', **metrics_dict}, ignore_index=True)



    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train_1_2_gram, y_train)
    y_pred3 = LRmodel.predict(X_test_1_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    metrics_dict = model_Evaluate(LRmodel, fpr, tpr, y_test, y_pred3)

    results_df = results_df.append({'Model': 'LogisticRegression', 'Dataset': 'X_train_1_2_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=49,
        n_jobs=-1
    )
    RFmodel.fit(X_train_1_2_gram, y_train)
    y_pred4 = RFmodel.predict(X_test_1_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred4)

    results_df = results_df.append({'Model': 'RandomForestClassifier1', 'Dataset': 'X_train_1_2_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_jobs=-1
    )
    RFmodel.fit(X_train_1_2_gram, y_train)
    y_pred5 = RFmodel.predict(X_test_1_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred5)

    results_df = results_df.append({'Model': 'RandomForestClassifier2', 'Dataset': 'X_train_1_2_gram', **metrics_dict}, ignore_index=True)



    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train_2_3_gram, y_train)
    y_pred1 = BNBmodel.predict(X_test_2_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    metrics_dict = model_Evaluate(BNBmodel, fpr, tpr, y_test, y_pred1)

    results_df = results_df.append({'Model': 'BernoulliNB', 'Dataset': 'X_train_2_3_gram', **metrics_dict}, ignore_index=True)



    SVCmodel = LinearSVC(dual='auto')
    SVCmodel.fit(X_train_2_3_gram, y_train)
    y_pred2 = SVCmodel.predict(X_test_2_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    metrics_dict = model_Evaluate(SVCmodel, fpr, tpr, y_test, y_pred2)

    results_df = results_df.append({'Model': 'LinearSVC', 'Dataset': 'X_train_2_3_gram', **metrics_dict}, ignore_index=True)



    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train_2_3_gram, y_train)
    y_pred3 = LRmodel.predict(X_test_2_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    metrics_dict = model_Evaluate(LRmodel, fpr, tpr, y_test, y_pred3)

    results_df = results_df.append({'Model': 'LogisticRegression', 'Dataset': 'X_train_2_3_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=49,
        n_jobs=-1
    )
    RFmodel.fit(X_train_2_3_gram, y_train)
    y_pred4 = RFmodel.predict(X_test_2_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred4)

    results_df = results_df.append({'Model': 'RandomForestClassifier1', 'Dataset': 'X_train_2_3_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_jobs=-1
    )
    RFmodel.fit(X_train_2_3_gram, y_train)
    y_pred5 = RFmodel.predict(X_test_2_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred5)

    results_df = results_df.append({'Model': 'RandomForestClassifier2', 'Dataset': 'X_train_2_3_gram', **metrics_dict}, ignore_index=True)



    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train_1_gram, y_train)
    y_pred1 = BNBmodel.predict(X_test_1_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    metrics_dict = model_Evaluate(BNBmodel, fpr, tpr, y_test, y_pred1)

    results_df = results_df.append({'Model': 'BernoulliNB', 'Dataset': 'X_train_1_gram', **metrics_dict}, ignore_index=True)



    SVCmodel = LinearSVC(dual='auto')
    SVCmodel.fit(X_train_1_gram, y_train)
    y_pred2 = SVCmodel.predict(X_test_1_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    metrics_dict = model_Evaluate(SVCmodel, fpr, tpr, y_test, y_pred2)

    results_df = results_df.append({'Model': 'LinearSVC', 'Dataset': 'X_train_1_gram', **metrics_dict}, ignore_index=True)



    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train_1_gram, y_train)
    y_pred3 = LRmodel.predict(X_test_1_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    metrics_dict = model_Evaluate(LRmodel, fpr, tpr, y_test, y_pred3)

    results_df = results_df.append({'Model': 'LogisticRegression', 'Dataset': 'X_train_1_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=49,
        n_jobs=-1
    )
    RFmodel.fit(X_train_1_gram, y_train)
    y_pred4 = RFmodel.predict(X_test_1_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred4)

    results_df = results_df.append({'Model': 'RandomForestClassifier1', 'Dataset': 'X_train_1_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_jobs=-1
    )
    RFmodel.fit(X_train_1_gram, y_train)
    y_pred5 = RFmodel.predict(X_test_1_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred5)

    results_df = results_df.append({'Model': 'RandomForestClassifier2', 'Dataset': 'X_train_1_gram', **metrics_dict}, ignore_index=True)



    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train_2_gram, y_train)
    y_pred1 = BNBmodel.predict(X_test_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    metrics_dict = model_Evaluate(BNBmodel, fpr, tpr, y_test, y_pred1)

    results_df = results_df.append({'Model': 'BernoulliNB', 'Dataset': 'X_train_2_gram', **metrics_dict}, ignore_index=True)



    SVCmodel = LinearSVC(dual='auto')
    SVCmodel.fit(X_train_2_gram, y_train)
    y_pred2 = SVCmodel.predict(X_test_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    metrics_dict = model_Evaluate(SVCmodel, fpr, tpr, y_test, y_pred2)

    results_df = results_df.append({'Model': 'LinearSVC', 'Dataset': 'X_train_2_gram', **metrics_dict}, ignore_index=True)



    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train_2_gram, y_train)
    y_pred3 = LRmodel.predict(X_test_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    metrics_dict = model_Evaluate(LRmodel, fpr, tpr, y_test, y_pred3)

    results_df = results_df.append({'Model': 'LogisticRegression', 'Dataset': 'X_train_2_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=49,
        n_jobs=-1
    )
    RFmodel.fit(X_train_2_gram, y_train)
    y_pred4 = RFmodel.predict(X_test_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred4)

    results_df = results_df.append({'Model': 'RandomForestClassifier1', 'Dataset': 'X_train_2_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_jobs=-1
    )
    RFmodel.fit(X_train_2_gram, y_train)
    y_pred5 = RFmodel.predict(X_test_2_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred5)

    results_df = results_df.append({'Model': 'RandomForestClassifier2', 'Dataset': 'X_train_2_gram', **metrics_dict}, ignore_index=True)



    BNBmodel = BernoulliNB()
    BNBmodel.fit(X_train_3_gram, y_train)
    y_pred1 = BNBmodel.predict(X_test_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    metrics_dict = model_Evaluate(BNBmodel, fpr, tpr, y_test, y_pred1)

    results_df = results_df.append({'Model': 'BernoulliNB', 'Dataset': 'X_train_3_gram', **metrics_dict}, ignore_index=True)



    SVCmodel = LinearSVC(dual='auto')
    SVCmodel.fit(X_train_3_gram, y_train)
    y_pred2 = SVCmodel.predict(X_test_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
    metrics_dict = model_Evaluate(SVCmodel, fpr, tpr, y_test, y_pred2)

    results_df = results_df.append({'Model': 'LinearSVC', 'Dataset': 'X_train_3_gram', **metrics_dict}, ignore_index=True)



    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(X_train_3_gram, y_train)
    y_pred3 = LRmodel.predict(X_test_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
    metrics_dict = model_Evaluate(LRmodel, fpr, tpr, y_test, y_pred3)

    results_df = results_df.append({'Model': 'LogisticRegression', 'Dataset': 'X_train_3_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=49,
        n_jobs=-1
    )
    RFmodel.fit(X_train_3_gram, y_train)
    y_pred4 = RFmodel.predict(X_test_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred4)

    results_df = results_df.append({'Model': 'RandomForestClassifier1', 'Dataset': 'X_train_3_gram', **metrics_dict}, ignore_index=True)



    RFmodel = RandomForestClassifier(
        n_jobs=-1
    )
    RFmodel.fit(X_train_3_gram, y_train)
    y_pred5 = RFmodel.predict(X_test_3_gram)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
    metrics_dict = model_Evaluate(RFmodel, fpr, tpr, y_test, y_pred5)

    results_df = results_df.append({'Model': 'RandomForestClassifier2', 'Dataset': 'X_train_3_gram', **metrics_dict}, ignore_index=True)


    results_df.to_csv('../data/results/evaluation_results.csv', index=False)
    # display(results_df)


    # display(results_df.sort_values(by='Accuracy', ascending=False))