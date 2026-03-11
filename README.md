\# Movie Review Sentiment Analysis



This project implements a sentiment analysis model to classify movie reviews as positive or negative. It uses a \*\*Logistic Regression\*\* classifier and a custom NLP preprocessing pipeline.



Project Structure

\- `main.py`: The main script to run the training and evaluation.

\- `text\_processing\_functions.py`: Modular functions for text cleaning and model optimization.

\- `Movie\_Sentiment\_Analysis.ipynb`: Exploratory data analysis and initial experiments.

\- `requirements.txt`: List of Python dependencies.



Key Features

\- \*\*Contraction Expansion\*\*: "don't" -> "do not" using the `contractions` library.

\- \*\*Text Normalization\*\*: Accented character removal via `unidecode` and punctuation stripping.

\- \*\*Hyperparameter Tuning\*\*: Automated search for the best `C` value in Logistic Regression.

\- \*\*Feature Importance\*\*: Extraction of the top 30 tokens that drive sentiment prediction.



How to Run

1\. Install dependencies:

&#x20;  pip install -r requirements.txt





\*\*Initial Idea\*\*:

The dataset is described here: https://www.aclweb.org/anthology/P04-1035.pdf



It is part of nltk, so it is convenient for us to use.



The goal is to build a first machine learning model using the tools that we have seen so far: choose how to preprocess the text, create a bag of words feature representation, train a model using an ML method of your choice.



You need to use the following split for the data:



test: 30% of the documents

The rest of the documents will be split as

train: 75% of the documents

validation: 25% of the documents

Use accuracy as evaluation measure.

