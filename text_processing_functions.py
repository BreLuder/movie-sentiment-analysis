import spacy
import string
import contractions
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

print("Libreria caricata")
nlp_en = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
print("modello caricato")


def txt_polishing(corpus, stop_words):
  elaborated_texts=[]
  for text in corpus:
    expanded_text = contractions.fix(text) # Expansion of contractions didn't-> did not
    text = expanded_text.split()

    cleaned_words = []
    for word in text:
      clean_word = word.lower()   #Lowercasing words
      clean_word = "".join([i for i in clean_word if i not in string.punctuation]) #Remove punctuation
      clean_word = unidecode(clean_word) #Normalize accented characters

      if clean_word and clean_word.isalpha() and clean_word not in stop_words: #Filtered removing stopwords
          cleaned_words.append(clean_word)
    elaborated_texts.append(" ".join(cleaned_words))
  return elaborated_texts


def text_polishing(corpus,stop_words):
    elaborated_texts = []
    for text in corpus:
        #Expanding conctractions
        expanded_text = contractions.fix(text)

        #Transformation into doc object.
        doc = nlp_en(expanded_text)

        cleaned_words = []
        for token in doc:
            if token.pos_ == "PROPN": #Eliminating Proper Nouns from the analysis
                continue

            #Tokenization, characters standardization, elimination of punctuation.
            clean_word = token.lemma_.lower()
            clean_word = unidecode(clean_word)
            clean_word = "".join([i for i in clean_word if i not in string.punctuation])

            #Elimination of words containing numbers.
            if clean_word and clean_word.isalpha() and clean_word not in stop_words:
                cleaned_words.append(clean_word)

        elaborated_texts.append(" ".join(cleaned_words))
    return elaborated_texts




def top_tokens_extraction(vectorizer, final_clf):
  feature_names = vectorizer.get_feature_names_out()

  model_weights = final_clf.coef_[0]

  word_features = list(zip(model_weights, feature_names))
  ordered_words = sorted(word_features, key=lambda x: abs(x[0]), reverse=True)[:30]
  return ordered_words

def optimal_model(X_train_vec, y_train, X_val_vec, y_val):
  best_acc = 0
  best_c=0
  best_model = None
  
  for c_val in [0.01, 0.1, 0.15, 0.5, 1, 10]:
      clf = LogisticRegression(C=c_val, random_state=42, max_iter=1000)
      clf.fit(X_train_vec, y_train)

      score = accuracy_score(y_val, clf.predict(X_val_vec))
      if score > best_acc:
          best_acc = score
          best_c = c_val
          best_model = clf
  return best_c, best_model