from django.shortcuts import render

# Create your views here.
# chatbot/views.py

from rest_framework.response import Response
from rest_framework.decorators import api_view
from .chatbot_logic import classify_intent_input, generate_response

@api_view(['POST'])
def chatbot_view(request):
    user_input = request.data.get('user_input', '')  # Assuming user input is sent in the 'user_input' field
    if user_input.lower() == "exit":
        response = {"chatbot_response": "Goodbye!"}
    else:
        intent = classify_intent_input(user_input)
        response_text = generate_response(user_input, intent,)
        response = {"chatbot_response": response_text}
    return Response(response)


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import pickle
# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('C:\\Users\\Lenovo\\project\\fine_tuned_bert_automotive')
# Load the model and tokenizer from pickle files
#with open('C:\\Data\\chatbot_project\\chatbot\\fine_tuned_bert_automotive.pkl', 'rb') as model_file:
    #model = pickle.load(model_file)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define your label map here (example)
#label_map = {0: 'greeting', 1: 'question', 2: 'complaint'}
unique_labels=['car brands',
 'car models - coupe info',
 'greetings',
 'car maintenance',
 'car models - hatchback info',
 'car models - truck info',
 'car models - SUV info',
 'car models - sedan info',
 'car models',
 'car features']
# Create a label map
label_map = {label: index for index, label in enumerate(unique_labels)}
def classify_intent(user_input):
    inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_intent_index = torch.argmax(probabilities).item()
    return predicted_intent_index


def decode_intent(intent_index):
    print(label_map)
    label_map_reverse = {v: k for k, v in label_map.items()}
    print(label_map_reverse)
    return label_map_reverse[intent_index]


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

@method_decorator(csrf_exempt, name='dispatch')
def classify_intent_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data.get('user_input')
        print(user_input)
        if not user_input:
            return JsonResponse({'error': 'No user input provided'}, status=400)

        intent_index = classify_intent(user_input)
        print(intent_index)
        intent_label = decode_intent(intent_index)

        # Generate response based on intent label
        response = generate_response1(user_input, intent_label, response_templates)

        return JsonResponse({'intent': intent_label, 'response': response})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


# Example response generation function
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



def generate_response1(user_input, intent_label, response_templates):
    return response_templates.get(intent_label, 'I\'m not sure how to respond to that.')
