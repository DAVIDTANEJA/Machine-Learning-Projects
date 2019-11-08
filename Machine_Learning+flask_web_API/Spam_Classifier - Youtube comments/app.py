from flask import Flask, render_template, url_for, request

import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

import re                             # Data cleaning removing stopwords(, . etc.)
import nltk                           # and creating Bag of Words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    # df = pd.read_csv("YoutubeSpamMergedData.csv", encoding="ISO-8859-1")      # this dataset is in binary, using : encoding 
    # dataset = df.iloc[:,5:7]   # Now we have 'CONTENT' and 'CLASS'  columns only.
    # dataset.to_csv('SpamModel.csv', index=None)         # saving the dataset into csv file, earlier 1 is in binary, using 2 columns.

    df1 = pd.read_csv('SpamModel.csv')

    # Data Cleaning
    ps = PorterStemmer()
    corpus=[]
    for i in range(0, len(df1)):
        review = re.sub('[^a-zA-Z]', ' ', df1['CONTENT'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    # Creating Bag of words (Document matrix)
    cv = CountVectorizer(max_features=5000) 

    X = cv.fit_transform(corpus).toarray()
    y = df1.iloc[:,1].values     

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = MultinomialNB()
    model.fit(X_train, y_train) 
    acc = math.ceil(model.score(X_test, y_test)*100)      # math.ceil used to round the decimals upto 2 
    

    # joblib.dump(model, 'youtubeSpam_model')    # dumping the model using joblib
    # model = joblib.load('youtubeSpam_model')

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('result.html', prediction = my_prediction, data=comment, accuracy=acc)


if __name__ == "__main__":
    app.run(debug=True)