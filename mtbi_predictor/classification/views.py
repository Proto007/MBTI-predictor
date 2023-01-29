import pandas as pd
from rest_framework import status, viewsets
from rest_framework.response import Response

from .models import *
from .serializers import *


class TrainModelViewset(viewsets.ModelViewSet):
    queryset = TrainModel.objects.all()
    serializer_class = TrainModelSerializer


class ClassifyViewset(viewsets.ModelViewSet):
    queryset = Classify.objects.all()
    serializer_class = ClassifySerializer
