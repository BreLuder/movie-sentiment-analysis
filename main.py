
import random
import spacy
from nltk.corpus import movie_reviews
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from text_processing_functions import text_polishing, optimal_model, top_tokens_extraction


import nltk
nltk.download('movie_reviews') # loads the dataset
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

custom_words = {'plot', 'movie', 'film', 'actor', 'scene', 'director', 'watch'}
stop_words.update(custom_words)

nlp_en = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

documents = [(movie_reviews.raw(fileid), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

print("\nNumber of docs loaded:", len(documents))

corpus_raw = [ x[0] for x in documents ]  # corpus_raw is a list of strings (reviews) to be converted into vectors
y_corpus = [ x[1] for x in documents ]    # y_corpus are the sentiment labels of the reviews (nothing to be done here)


random.seed(42)

#Data splitting.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    corpus_raw, y_corpus, test_size=0.3, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

#BoW vectorization
#vectorizer = CountVectorizer(stop_words='english', max_features=5000)
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=5000, ngram_range=(1, 2))

#Pre-Processing of training and validation sets.
train_polished = text_polishing(X_train, stop_words)
val_polished = text_polishing(X_val, stop_words)

#Vectorization (Dictionary creation) of training and validation sets.
X_train_vec = vectorizer.fit_transform(train_polished)
X_val_vec = vectorizer.transform(val_polished)


#Hyper-parameter optimization.
best_c, best_model = optimal_model(X_train_vec, y_train, X_val_vec, y_val)
print(f"Optimal C found: {best_c}")


#Re-union of training and validation sets. Same with labels
#Final processing, vectorization and evaluation on test set.

full_train_polished = train_polished + val_polished
y_train_full_list = list(y_train) + list(y_val)

test_polished = text_polishing(X_test, stop_words)

#Vectorization using TF-IDF considering bi-grams.
vectorizer_final = TfidfVectorizer(stop_words=list(stop_words), max_features=5000, ngram_range=(1, 2))

#Polishing of the final training set and the test set.
X_train_full_vec = vectorizer_final.fit_transform(full_train_polished)
X_test_vec_final = vectorizer_final.transform(test_polished)

#Final model
final_clf = LogisticRegression(C=best_c, random_state=42, max_iter=1000)
final_clf.fit(X_train_full_vec, y_train_full_list)

y_pred = final_clf.predict(X_test_vec_final)

print(f"Final Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Detailed Performance Report ---")
print(classification_report(y_test, y_pred))

print("Top 30 tokens by absolute parameter value:")
top_tokens = top_tokens_extraction(vectorizer_final, final_clf)
for weight, word in top_tokens:
    print(f"{word:<15} | weight: {weight:.4f}")