from django.contrib import admin
from django.urls import path,include
from .views import ML,DL,Home

urlpatterns = [
    path('house/',ML.house_predication,name='ml1'),
    path('car/',DL.car_classification,name="car"),
    path('home/',Home,name="home"),
    path('',ML.sms_spam_classifier,name='sms_spam'),
]