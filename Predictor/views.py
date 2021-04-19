from django.shortcuts import render
from rest_framework.views import APIView,status
from rest_framework.response import Response
from django.http import JsonResponse
from Predictor import prediction
import json

# Create your views here.

class PredictionAPI(APIView):

    def get(self,request):

        if request.method == "GET":

            ticker = request.GET.get('ticker')

            result = prediction.import_data(ticker)

            #result = json.dumps(str(result))

            response  = { 'prediction': result}

            return Response(str(result), status=status.HTTP_200_OK)

    def post(self):

        pass




