import pickle
import  numpy as np
import pandas as pd
import random
# Define response templates for each intent
response_templates = {
        'car models': "Our available models include sedan, SUV, coupe, hatchback, and truck.",
        'car brands': "We offer vehicles from top brands such as Toyota, Honda, Ford, Chevrolet, and Tesla.",
        'car features': "Our vehicles come with a range of features including engine options, transmission types, fuel efficiency, and safety features.",
        'car maintenance': "Our maintenance services include oil change, tire rotation, brake inspection, battery check, and fluid level check.",
        'greetings': "Hello! How can I assist you today?"
    }

# Add response for each specific car model
response_templates['car models - sedan info'] = "Our sedan models offer a perfect combination of style, comfort, and performance. Explore our range of sedan models for the latest in technology, safety, and luxury features."
response_templates['car models - SUV info'] = "Our SUV lineup includes versatile models designed for all your adventures. Discover spacious interiors, advanced safety features, and powerful performance in our SUVs."
response_templates['car models - coupe info'] = "Experience the thrill of driving with our coupe models. With sleek designs and powerful engines, our coupes deliver an exhilarating performance."
response_templates['car models - hatchback info'] = "Our hatchback models offer practicality and versatility in a compact package. Enjoy agile handling, ample cargo space, and fuel-efficient engines in our hatchbacks."
response_templates['car models - truck info'] = "Get the job done with our rugged and reliable truck models. From hauling heavy loads to off-road adventures, our trucks are built to tackle any task with ease."




def classify_intent_input(user_input):
    with open(r'C:\Data\chatbot_project\chatbot\neivebayesclf.pkl', "rb") as file:
        clf, tfidf_vectorizer = pickle.load(file)
    user_vector = tfidf_vectorizer.transform([user_input])
    #predicted_probabilities = clf.predict_proba(inv)[0]
    #predicted_intent = clf.classes_[predicted_probabilities.argmax()]
    predicted_intent = clf.predict(user_vector)[0]
    return predicted_intent

"""
def model_build(intents):
    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(intents['data'])
    clf = MultinomialNB()
    clf.fit(X_train, intents['target'])
    return clf, tfidf_vectorizer
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.naive_bayes import MultinomialNB
import string
import random

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to perform lemmatization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

# Function to preprocess user input
def preprocess_input(user_input):
    # Tokenize
    tokens = word_tokenize(user_input.lower())
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize
    lemmatized_tokens = lemmatize_text(tokens)
    return ' '.join(lemmatized_tokens)
"""
# Function to calculate semantic similarity between user input and predefined responses
def semantic_similarity(user_input, responses):
    tfidf_vectorizer = TfidfVectorizer()
    response_vectors = tfidf_vectorizer.fit_transform(responses)
    user_vector = tfidf_vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, response_vectors)
    return similarity_scores
"""
# Function to generate response
def generate_response(user_input, intent, knowledge_base={}):
    user_input = preprocess_input(user_input)
    print("user preprocess:",user_input)

    # If the predicted intent is consistently "car brands" for new unseen inputs, handle it as a fallback

    # Check for specific keywords or patterns in the input to determine if it's unrelated to car brands
    unrelated_keywords = ["weather", "news", "sports", "music"]
    unrelated_keywords1 = ["suv","coupe","hatchback","truck","sedan","car model", "car brands","Toyota","Honda","Ford","Chevrolet","Tesla","car feature","feature", "car maintenance","maintenance","brand","model","Toyota", "Honda", "Ford", "Chevrolet","Tesla"]
    #unrelated_keywords2 = ["Our", "maintenance", "services", "oil change", "tire rotation", "brake inspection", "battery check", fluid level check]



    # Get responses based on intent
    responses = response_templates.get(intent, [])
    print("response:",responses)
     # Check for greetings
    greetings1 = ["hi", "hello", "hey", "hola", "howdy","greeting"]
    if intent == "greetings" and any(word in user_input for word in greetings1):
        return responses

    elif intent == "car maintenance":
      print(user_input)
      if user_input  in unrelated_keywords1:
        return responses
    elif intent=='car models - sedan info' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car models - SUV info' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car models - coupe info' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car models - truck info' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car models - hatchback info' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car features' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car models' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car brands' and any(word in user_input for word in unrelated_keywords1):
      return responses
    elif intent=='car maintenance' and any(word in user_input for word in unrelated_keywords1):
      return responses
    else:
        return "It seems like you're asking about something unrelated to automotive industry. Can you please provide more information or ask a different question?"

    if responses == None or "null":
        return "Sorry, I don't have information about that. Can you rephrase your question?"