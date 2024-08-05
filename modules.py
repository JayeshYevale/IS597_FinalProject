import time
from datetime import timedelta # for using time library
import numpy as np # numpy for computation
import pandas as pd  #pandas library for computation
import os.path #for path directory
from nltk.stem import PorterStemmer, WordNetLemmatizer  # for text stemming
from sklearn.model_selection import train_test_split    # for splitting the data
from sklearn.preprocessing import StandardScaler   # for scaling the data
from sklearn.tree import DecisionTreeClassifier  # for Decision tree classifier
from sklearn.linear_model import LogisticRegression # for logistic regression classifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC  # for support vector machine classifier
from sklearn.ensemble import RandomForestClassifier  # for random forest classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix  #for using confusion matrix
import matplotlib.pyplot as plt   # for displaying output visualization in matplotlib
import seaborn as sns   # for displaying output visualization in seaborn
from sklearn.metrics import ConfusionMatrixDisplay # for displaying confusion matrix
from sklearn.metrics import classification_report # for classification report
from sklearn.feature_extraction.text import TfidfVectorizer # import vectorizer for text-to-numeric conversion for sentiment analysis
import warnings
warnings.filterwarnings('ignore')
import csv # for working with CSV data


def load_data(in_filename, target):
    # Load data from dataset\n",
    df = pd.read_excel(in_filename)
    print("************** Loading Data ************")
    print("No of Rows: {}".format(df.shape[0]))
    print("No of Columns: {}".format(df.shape[1]))
    # Process and clean the exisiting data\n",
    df['A_PTYPE'] = df['A_PTYPE'].apply(lambda x: 1 if x == 1 else 0)
    df['A_DOA'] = df['A_DOA'].apply(lambda x: 1 if x == 1 else 0)
    df['A_WEATHER'] = df['A_WEATHER'].apply(lambda x: 1 if x == 1 else (3 if x == 99 else 2))
    df['A_PERINJ'] = df['A_PERINJ'].apply(lambda x: 0 if x == 6 else 1)
    df['A_HELMUSE'] = df['A_HELMUSE'].apply(lambda x: 0 if x == 3 else (1 if x in [1, 2] else 2))

    # Remove unnecessary columns\
    df = df.drop('ST_CASE', axis=1)

    print("************** Data Info ************")
    print(df.info())

    print("************** Data Values ************")
    print(df.nunique().sort_values())

    print("************** Information on 'Target' variable ************")
    counts = df[target].value_counts()
    original_labels = counts.index.tolist()
    sizes = counts.values
    label_mapping = {0: 'No Deaths', 1: 'Deaths'}
    new_labels = [label_mapping[label] for label in original_labels]
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=new_labels, autopct='%1.1f%%', startangle=30, colors=['#ff9999', '#66b3ff'])
    plt.title('Pie Chart of Binary Column')
    plt.show()
    print("************** Distribution on other variables ************")
    for column in df.columns:
        if df[column].dtype == 'object' or not len(df[column].unique()) > 10:  # Consider as categorical if dtype is 'object' or has <= 10 unique values
            plt.figure(figsize=(4, 4))
            sns.countplot(x=column, data=df)
            plt.title(f'Bar Chart of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    print("************** Crosstab Heatmap for dataset ************")
    plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=True, cmap='viridis')
    plt.show()

    #Seperate X and y for further processing
    X = df.drop(target, axis=1)
    y = df[target]

    # Split the data in train and test\n",
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

    # Scaling the X data for processing\n",
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_train, scaled_X_test, y_train, y_test


def fit_model(X_train, y_train, model):
    """
      Model fitting with options of classifiers:
      decision tree, svm, knn, naive bayes, random forest, and gradient boosting
      param X_train: X train data
      param y_train: y train data,
      param model: name of classifier
    """
    print("\n ************** Training Model: " + model+ " **************")
    if model=='DT':
        DT = DecisionTreeClassifier(max_depth=2)
        model = DT.fit(X_train, y_train)
    elif model=='SVM':
        SVM = SVC(kernel='linear', probability=True)
        model = SVM.fit(X_train, y_train)
    elif model=='LR':
        LR = LogisticRegressionCV(cv=10,random_state=50, verbose=1)
        model = LR.fit(X_train, y_train)
    elif model=='RF':
        RF = RandomForestClassifier(max_depth=2, random_state=0)
        model = RF.fit(X_train, y_train)
    elif model=='GB':
        GB = GradientBoostingClassifier()
        model = GB.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred, eval_model):
    """
      evaluate model performance
      param y_test: y test data
      param y_pred: t prediction score
      param eval_model: indicator if this function is on or off
    """
    if eval_model:
        print('\n************** Model Evaluation **************')
        print('\n Confusion Matrix: \n')
        print(confusion_matrix(y_test, y_pred))
        print('\n Classification Report: \n')
        print(classification_report(y_test, y_pred, digits=4))

