from django.shortcuts import render
from rest_framework.views import APIView,status
from rest_framework.response import Response
from django.http import JsonResponse
from Predictor import prediction, model_building
import json

# Create your views here.

class PredictionAPI(APIView):

    def get(self,request):

        if request.method == "GET":

            ticker = request.GET.get('ticker')

            result = prediction.predict(ticker)

            return Response(str(result), status=status.HTTP_200_OK)

    def post(self):

        pass

class ModelBuild(APIView):

    def get(self, request):

        model_building.build_model()

        res = {
            'Response': "Model Build Finished"
        }

        return Response(res, status=status.HTTP_200_OK)

    def post(self, request):

        pass




