import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score
import pickle
from binary_classifier import y_test, X_test

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    prediction_test_knn = model.predict(X_test)
    output = accuracy_score(y_test, prediction_test_knn)
    return render_template('index.html', prediction_text='KNN Accuracy for test data (self accuracy): {}%'.format(round(output*100),3))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)