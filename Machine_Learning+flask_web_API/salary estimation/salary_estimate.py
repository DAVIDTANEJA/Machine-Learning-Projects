import pandas as pd
from word2number import w2n
import math
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

df = pd.read_csv('hiring.csv')      # reading the dataset

df['experience'] = df['experience'].fillna('Zero')   # 1st convert all NaN values into 'Zero'
df['experience'] = df['experience'].apply(w2n.word_to_num)   # 2nd convert word to number

df['test_score'] = df['test_score'].fillna(math.floor(df.test_score.mean()))   # fill NaN of 'test_score' with mean using math.floor()

# Dividing the dataset
X = df[['experience', 'test_score', 'interview_score']]
y = df['salary']

# Creating linear regression model
model = LinearRegression()
model.fit(X,y)

# Saving the model using joblib
joblib.dump(model, 'salary_estimation')          # dump the model into 'salary_estimation' file 
salary_model = joblib.load('salary_estimation')  # load the model