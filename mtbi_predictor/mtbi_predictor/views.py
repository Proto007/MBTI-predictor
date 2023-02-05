import requests
from django.shortcuts import render
from django.urls import reverse
from rest_framework.views import APIView


class Predict(APIView):
    """
    Handle requests from `homepage` template
    """

    def get(self, request):
        return render(request, "homepage.html")

    def post(self, request):
        query_text = str(request.data["query_text"])
        using_default = bool(request.data.get("using_default", False))
        response = requests.post(
            request.build_absolute_uri(reverse("classification:classify-list")),
            json={"query_text": query_text, "use_default": using_default},
        )
        if response.status_code == 200:
            return render(
                request, "homepage.html", {"prediction": response.json()["prediction"]}
            )
        return render(request, "homepage.html")
