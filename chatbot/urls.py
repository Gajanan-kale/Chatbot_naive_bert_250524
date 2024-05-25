# chatbot/urls.py

from django.urls import path
from .views import chatbot_view
from .views import classify_intent_view

urlpatterns = [
    path('chatbot/', chatbot_view, name='chatbot'),
    path('classify/', classify_intent_view, name='classify_intent'),
]
