# Binary Classification

## Overview
This project consists of creating a binary classifier using multiple binary classification algorithms to assess accuracy.

## Steps
* Cleaning the data set.
* Encoding the data.
* Testing algorithms

## Observation
- The best accuracy achieved on test data was from KNN algorithm at 61%.
## How To Use
* It is better to create a [virtual](https://docs.python.org/3/library/venv.html) environment.
* Install pandas, numpy and scikit-learn or run ```pip install -r requirements.txt```.
* Navigate to the directory containing the files and exectute 
```py binay_classifier.py```.
## Flask
- By importing pickle we can save our model to server our results through Flask.
* Run ```py app.py```.
* Visit [http://localhost:5000/](http://localhost:5000/).
