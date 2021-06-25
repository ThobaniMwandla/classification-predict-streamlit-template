"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import re
import nltk
import joblib,os
import streamlit as st
import streamlit.components.v1 as components

#import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import warnings

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings(action = 'ignore') 

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def display_prediction(input_text):
    if input_text[0]==-1:
        output="Anti"
        st.error(f"{output} Sentiment Predicted")
        st.error('Tweets that do not support the belief of man-made climate change.')
    elif input_text[0]==0:
        output="Neutral"
        st.info(f"{output} Sentiment Predicted")
        st.info("Tweets that neither support nor refuse beliefs of climate change.")
    elif input_text[0]==1:
        output ="Pro"
        st.success(f"{output} Sentiment Predicted")
        st.success("Tweets that support the belief of man-made climate change")
    else:
        output = "News"
        st.warning(f"{output} Sentiment Predicted")
        st.warning("Tweets linked to factual news about climate change.")
    
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About The App", "Prediction", "Data Visualisation", "Model Performance"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "About The App":
		st.image('resources/imgs/changes.gif', caption='Climate Change',use_column_width=True)
		st.subheader("About the App")
		st.info("The entire app is built using Machine Learning models that are able to classify whether or not a person believes in climate change, based on their novel tweet data.")
		# You can read a markdown file from supporting resources folder
		st.markdown("Below is just the small portion of dataset that has been used to train the models")

		st.subheader("Introduction")
		st.markdown(
			"""It is undeniable that climate change is one of the most talked topics of our times and one of the biggest challenges the world is facing today. 
			In the past few years, we have seen a steep rise on the Earth's temperature, causing a spike in wild fires, drought, rise of sea levels due to melting glaciers, rainfall pattern shifts, flood disasters.
			"""
			)

		st.subheader("Why App Was Created ?")
		st.markdown('The aim of this App is to gauge the public perception of climate change using twitter data')

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	if selection == "Prediction":
		# Creating a text box for user input
		models_used = ["LogisticRegression", "RidgeClassifier", "LinearSVC", "SGDClassifier", "Support Vector Machine"]
		selected_model = st.selectbox('Choose Your Favourite Model', models_used)

		if selected_model =="RidgeClassifier":
				st.subheader('Model Info Below')
				st.info("This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case)")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/RidgeClassifier.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model =="SGDClassifier":
				st.subheader('Model Info Below')
				st.info("This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate)")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/SGDClassifier.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model == "LinearSVC":
				st.subheader('Model Info Below')
				st.info("This Classifier fit to the data you provide, returning a `best fit` hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the `predicted` class is.")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model =="Support Vector Machine":
				st.subheader('Model Info Below')
				st.info("This Classifier finds a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/Support_Vector_Machine.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model =="LogisticRegression":
				st.subheader('Model Info Below')
				st.info("Logistic regression models the probabilities for classification problems with two possible outcomes. It's an extension of the linear regression model for classification problems.")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)
		
	if selection == "Data Visualisation":
		st.image('resources/imgs/tweet_dst.png', caption='Climate Change',use_column_width=True)
		st.info("From the above Tweet vs Sentiment charts, approximately 54% of the tweets on our data believe that cliamte change is real and whereas only 8% are in opposition. However, 15% of the tweets are neither negative or positive- meaning they are neutral. Appearing second after 'Pro' tweets is 'News' tweets at 23%- very indicative of the relevancy of the topic.")
		st.subheader("Mostly used words are show below")
		st.image('resources/imgs/mostly_used.png', caption='Climate Change',use_column_width=True)
		st.info(
			"""
			The mostly common used words, in hierachy, across all four categories are 'climate', 'change', 'global', and then 'warming'. This is expected as the topic is centred around these words.

			Also, words like 'science' and 'scientist' are frequent- which could imply that people are tweeting about scientific studies that support their views on climate change.

			However, words such as 'hoax', 'fake', 'left' and 'scam' on ANTI tweets are more emphasized than the other categories- which is inline with the denier's sentiment.
			"""
		)

	if selection == "Model Performance":
		st.info("Below is the picture of how each Model performs")
		st.image('resources/imgs/models.png', caption='Climate Change',use_column_width=True)
		model_selected = ["Select Model", "LogisticRegression", "RidgeClassifier", "LinearSVC", "SGDClassifier", "SVC"]
		selected_model = st.selectbox("Choose Model Metrics By Model Type", model_selected)
		
		if selected_model =="LinearSVC":
			st.markdown("<h3 style='color:#0069d1'>Model Performance</h3><br/>",unsafe_allow_html=True)
			components.html(
				"""
				<!DOCTYPE html>
				<html lang="en">
				<head>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
				<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
				<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
				</head>
				<body>

				<div class="container">          
				<table class="table">
					<thead>
						<tr style="text-align: right">
						<th></th>
						<th>Vectorizer Type</th>
						<th>Model Name</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1-score</th>
						<th>Execution Time</th>
						</tr>					
					</thead>
					<tbody>
						<tr>
                        <th></th>
                        <td>TF_1</td>
                        <td>LinearSVC</td>
                        <td>71.7736</td>
                        <td>72.8192</td>
                        <td>72.2926</td>
                        <td>0.250359</td>
                        </tr>
					</tbody>
				</table>
				</div>

				</body>
				</html>
				
				"""
			)
			
		if selected_model =="SVC":
			st.markdown("<h3 style='color:#0069d1'>Model Performance</h3><br/>",unsafe_allow_html=True)
			components.html(
				"""
				<!DOCTYPE html>
				<html lang="en">
				<head>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
				<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
				<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
				</head>
				<body>

				<div class="container">          
				<table class="table">
					<thead>
						<tr style="text-align: right">
						<th></th>
						<th>Vectorizer Type</th>
						<th>Model Name</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1-score</th>
						<th>Execution Time</th>
						</tr>					
					</thead>
					<tbody>
						<tr>
                        <th></th>
                        <td>TF_1</td>
                        <td>SVC</td>
                        <td>76.8198</td>
                        <td>75.2212</td>
                        <td>76.0121</td>
                        <td>47.981690</td>
                        </tr>
					</tbody>
				</table>
				</div>

				</body>
				</html>
				"""
			)

		if selected_model =="RidgeClassifier":
			st.markdown("<h3 style='color:#0069d1'>Model Performance</h3><br/>",unsafe_allow_html=True)
			components.html(
				"""
				<!DOCTYPE html>
				<html lang="en">
				<head>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
				<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
				<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
				</head>
				<body>

				<div class="container">          
				<table class="table">
					<thead>
						<tr style="text-align: right">
						<th></th>
						<th>Vectorizer Type</th>
						<th>Model Name</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1-score</th>
						<th>Execution Time</th>
						</tr>					
					</thead>
					<tbody>
						<tr>
                        <th></th>
                        <td>TF_1</td>
                        <td>RidgeClassifier</td>
                        <td>64.7715</td>
                        <td>65.9924</td>
                        <td>65.3763</td>
                        <td>4.340930</td>
                        </tr>
					</tbody>
				</table>
				</div>

				</body>
				</html>
				"""
			)

		if selected_model =="LogisticRegression":
			st.markdown("<h3 style='color:#0069d1'>Model Performance</h3><br/>",unsafe_allow_html=True)
			components.html(
				"""
				<!DOCTYPE html>
				<html lang="en">
				<head>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
				<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
				<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
				</head>
				<body>

				<div class="container">          
				<table class="table">
					<thead>
						<tr style="text-align: right">
						<th></th>
						<th>Vectorizer Type</th>
						<th>Model Name</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1-score</th>
						<th>Execution Time</th>
						</tr>					
					</thead>
					<tbody>
						<tr>
                        <th></th>
                        <td>TF_1</td>
                        <td>LogisiticRegression</td>
                        <td>76.8198</td>
                        <td>75.2212</td>
                        <td>76.0121</td>
                        <td>47.981690</td>
                        </tr>
					</tbody>
				</table>
				</div>

				</body>
				</html>
				"""
			)

		if selected_model =="SGDClassifier":
			st.markdown("<h3 style='color:#0069d1'>Model Performance</h3><br/>",unsafe_allow_html=True)
			components.html(
				"""
				<!DOCTYPE html>
				<html lang="en">
				<head>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
				<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
				<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
				</head>
				<body>

				<div class="container">          
				<table class="table">
					<thead>
						<tr style="text-align: right">
						<th></th>
						<th>Vectorizer Type</th>
						<th>Model Name</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1-score</th>
						<th>Execution Time</th>
						</tr>					
					</thead>
					<tbody>
						<tr>
                        <th></th>
                        <td>TF_1</td>
                        <td>SGDClassifier</td>
                        <td>71.7736</td>
                        <td>72.8192</td>
                        <td>72.2926</td>
                        <td>0.250359</td>
                        </tr>
					</tbody>
				</table>
				</div>

				</body>
				</html>
				"""
			)

		st.subheader("You may click on the button below to jump to the conclusiion")
		if st.button("Click me to see the conclusion"):
				st.info(
					"""
					The SVC model performed better compared to all the models tested with an F1-Score of approximately 76%, when a Tfid vectorizer is used. After performing a hyperparameter tuning of the model, the F1-score rose by a very small margin of 0.5%.

					Naive Bayes process speed is the fastest, while it produced an underperforming average F1-score.

					The Logistic Regression and the LinearSVC is considerably fast and yet produces well above average F1-scores, with a very less margin from the best score. In loose terms, the two models are able to effortlessly produce quality performance, even though its F1-scores were not the best compared to the best score model.

					Whereas the SVC with the best f1-score takes a long time to produce the highest model. Loosely meaning it is trading processing speed over accuracy.

					Therefore, with this realization, overall, the Logistic regression, when the TF1 vectorizer type is used, performed better than all the models with an F1-score of approximately 74% and execution of 3 seconds. Even though the SVC, Tfid, produces better F1 results by a margin of 2%, the SVc is Approximately 15 times slower than Logistic Regression, when using the Tfid vectorizer type.

					However, with the aim of getting the best f1-score, we will select the SVC, Tfid vectorizer, as our best model
					"""
				)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
