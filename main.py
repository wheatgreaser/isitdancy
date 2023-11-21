import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, render_template, url_for, request
import time

app = Flask(__name__)

danceability = ""

global model, cv
def ml():
  df = pd.read_csv('data.csv', low_memory=False).head(10000)

  X = df['song_name']
  y = df['danceability']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

  cv = CountVectorizer()
  features = cv.fit_transform(X_train)
  model = linear_model.LinearRegression()

  model.fit(features, y_train)
  return model, cv

model, cv = ml()

def predict(song_name):
  
  d= (model.predict(cv.transform([song_name]))) 
  return d




@app.route('/', methods=['GET', 'POST'])
 
def page():
  

  if request.method == 'POST':
    dancy = predict(request.form.get("name"))
    if dancy > 0.75:
      danceability = "dancy"

    else:
      danceability = "non dancy"
    
    return render_template('index.html', danceability = danceability)
    
  return render_template('index.html', danceability = "")
    
    

if __name__ == '__main__':
  app.run(debug = True, host = '0.0.0.0')






