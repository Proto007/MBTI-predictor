from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import *

app_name = "classification"

router = DefaultRouter()
router.register("classification", ClassifyViewset, basename="classify")
router.register("train_model", TrainModelViewset, basename="train_model")

urlpatterns = [path("", include(router.urls))]
