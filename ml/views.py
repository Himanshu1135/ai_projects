from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
from .model_file.main import Model,predict_price
import json
from .form import ImageForm
from .models import Image
from .cnn_car_model.car_main import Model_car,predict
from .spam_classifier.sms_spam_main import SpamModel,Vectorizer,sms_spam_predict
# Create your views here.

def Home(request):
    if request.method =="POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    form = ImageForm()
    img = Image.objects.all()

    data = {
        "form":form,
        "img":img,
    }

    return render(request,"home.html",data)

class ML:

    @staticmethod
    @csrf_exempt
    def house_predication(request):
        if request.method == "POST":
            # -------------------------- Data input
            data = request.body.decode('utf-8')
            data = json.loads(data)
            location = data.get('location')
            bhk = data.get('bhk')
            area = data.get('area')
            bath = data.get('bath')
            #---------------------------- predication
            results = predict_price(location,area,bath,bhk,Model)
            ml_response = {"ML House Predication model response": results}
            
            return JsonResponse(ml_response,safe=False)
        

        return JsonResponse({'ml_response':"Hello"},safe=False)
    
    @staticmethod
    @csrf_exempt
    def sms_spam_classifier(request):
        if request.method == "POST":
            try:
                # -------------------------- Data input
                text = request.POST['input']
                # data = request.body.decode('utf-8')
                # data = json.loads(data)
                # text = data.get('sms')
                #---------------------------- predication   
                results = sms_spam_predict(text,SpamModel,Vectorizer)
                ml_response = {"result": results}
            except:
                ml_response = {"result": 'Found some Error'}

            return render(request,'sms.html',ml_response)

            
        #     return JsonResponse(ml_response,safe=False)
        

        return render(request,'sms.html')


class DL:

    @staticmethod
    @csrf_exempt
    def car_classification(request):
        if request.method == "POST":
            # # -------------------------- Data input
            data = request.body.decode('utf-8')
            data = json.loads(data)
            img_path = data.get('path')
            print(data)
            # #---------------------------- predication
            results = str(predict(img_path,Model_car))
            print(results)
            ml_response = {"DL model response": results}
            
            return JsonResponse(ml_response,safe=False)
        

        return JsonResponse({'Dl_response':"Hello"},safe=False)
