from rest_framework import viewsets

from .models import *
from .serializers import *


class ClassifyViewset(viewsets.ModelViewSet):
    """
    Viewset for Classify model
    """

    queryset = Classify.objects.all()
    serializer_class = ClassifySerializer
