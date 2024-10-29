from django.urls import path
from Moterole_API import views
from .views import PredictView



urlpatterns = [
    path('predict-view/', PredictView.as_view(), name='predict-view'),
]
