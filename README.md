# Resturant_Review_system

This project performs sentiment analysis on a dataset of reviews using a Naive Bayes classifier. The dataset consists of text reviews labeled as positive or negative sentiments.

# Files
untitled11.ipynb: Jupyter Notebook containing the code for sentiment analysis using a Naive Bayes classifier.
Reviews.tsv: Tab-separated values (TSV) file containing the dataset of reviews.
# Dependencies
*Python 3.x
*Jupyter Notebook
*pandas
*NumPy
*NLTK
*scikit-learn

Ensure that Reviews.tsv is in the project directory.
Run untitled11.ipynb to perform sentiment analysis using the Naive Bayes classifier.

# Dataset
The dataset (Reviews.tsv) contains a collection of reviews labeled with their corresponding sentiment (positive or negative).

# Approach
Preprocessing: The reviews are preprocessed by removing non-alphabetic characters, converting text to lowercase, tokenizing, stemming, and removing stopwords.
Feature Extraction: Bag of Words model using CountVectorizer is used to convert text data into numerical format.
Model Training: The Naive Bayes classifier is trained on the training set.
Prediction: The trained model is used to predict the sentiment of test reviews.
Evaluation: Confusion matrix, accuracy, and precision are calculated to evaluate the performance of the model.
# References
*NLTK Documentation: NLTK
*scikit-learn Documentation: scikit-learn
# Contributors
Nilkamal Basyak
