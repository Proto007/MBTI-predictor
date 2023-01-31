import pandas as pd
from rest_framework import status, viewsets
from rest_framework.response import Response

from .models import *
from .serializers import *
import shutil
import os
from mtbi_predictor.settings import BASE_DIR
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
import pickle
from sklearn.svm import LinearSVC



class TrainModelViewset(viewsets.ModelViewSet):
    queryset = TrainModel.objects.all()
    serializer_class = TrainModelSerializer

    def create(self, request):
        try:
            df = pd.read_csv(request.data["dataset"])
        except Exception:
            return Response({"message":"pandas failed to read file"}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
        preprocessed = request.data.get("preprocessed", "")
        TrainModel.objects.all().delete()
        if os.path.isdir(f'{BASE_DIR}/dataset'):
            shutil.rmtree(f'{BASE_DIR}/dataset')
        if not preprocessed:
            # TODO Preprocess
            pass
        train_x, test_x, train_y, test_y = model_selection.train_test_split(df['posts'],df['type'],test_size=0.2)
        tfidf = TfidfVectorizer()
        tfidf.fit_transform(df['posts'])
        train_x_tfidf = tfidf.transform(train_x)
        classifier = LinearSVC()
        classifier.fit(train_x_tfidf, train_y)
        text_classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LinearSVC())])
        text_classifier.fit(train_x, train_y)
        pickle.dump(text_classifier, open('new_svm_model', 'wb'))
        predictions = text_classifier.predict(test_x)
        accuracy = round(accuracy_score(test_y, predictions),2)
        TrainModel.objects.create(dataset=request.data["dataset"], preprocessed=bool(preprocessed), accuracy=accuracy)
        return Response(status=status.HTTP_200_OK)

class ClassifyViewset(viewsets.ModelViewSet):
    queryset = Classify.objects.all()
    serializer_class = ClassifySerializer
