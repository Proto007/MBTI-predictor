from django.shortcuts import render
from rest_framework.views import APIView


class Predict(APIView):
    """
    Handle requests from `homepage` template
    """

    def get(self, request):
        """
        Render `homepage.html` on get request
        """
        return render(request, "homepage.html")
