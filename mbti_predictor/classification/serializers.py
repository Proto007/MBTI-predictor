from rest_framework import serializers

from .models import *


class TrainModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainModel
        fields = ("dataset", "preprocessed", "accuracy")
        read_only_fields = ["accuracy"]


class ClassifySerializer(serializers.ModelSerializer):
    class Meta:
        model = Classify
        fields = ("query_text", "prediction", "use_default")
        read_only_fields = ["prediction"]
