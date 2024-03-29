from flask import Flask, request, render_template
import os
import pickle
import string
import nltk
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwords_En = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
# Modèle + Matrice de fonctionnalités TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

PEOPLE_FOLDER = os.path.join('static')

app = Flask(__name__)
model = pickle.load(open('model_final.pkl', 'rb'))

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict',methods=['POST'])
def predict():
    real_news_img = os.path.join(app.config['UPLOAD_FOLDER'], 'Real_news.png')
    fake_news_img = os.path.join(app.config['UPLOAD_FOLDER'], 'Fake_news.png')
    url = "https://fakers.azurewebsites.net/?news="
    nom = request.form["content"]
    reponse = requests.get(url + nom)
    prediction = reponse.text
    if prediction == '[1]':
        return render_template("index.html", user_image = fake_news_img)
    if prediction == '[0]':
        return render_template("index.html", user_image = real_news_img)
    else:
        pass

if __name__ == "__main__":
    app.run(debug=True)