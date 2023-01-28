from rest_framework.views import APIView
from django.shortcuts import render

class Predict(APIView):
    """
    Handle requests from `homepage` template
    """
    def get(self, request):
        """
        Render `homepage.html` on get request
        """
        return render(request, "homepage.html")
