import os
import pickle
import shutil

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rest_framework import status, viewsets
from rest_framework.response import Response
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from mbti_predictor.settings import BASE_DIR

from .models import *
from .serializers import *

nltk.download("stopwords")


def get_pos_tag(word: str) -> str:
    """
    Returns the pos tag of given word in a format that is recognized by wordnet lemmatizer

    Reference: https://github.com/Proto007/Topic-Modeling-Book-Descriptions/blob/404f4700b73fbcedec55136a85f68ebcc21580f0/backend/services/model/dataPreprocess.py#L45

    @params:
        word (string): query word
    @return
        pos_tag (string): pos tag that is recognizable by wordnet lemmatizer/ empty if the given word is empty or not one of the lemmatized pos
    """
    # Return empty string if query word is invalid
    if not word:
        return ""
    # Get the uppercase of first letter of the pos recognized by nltk's pos_tag function
    tag = nltk.pos_tag([word])[0][1][0].upper()

    # Return approprite wordnet pos tag based on the pag recognized by nltk's pos_tag function
    if tag == "J":
        return wordnet.ADJ
    elif tag == "N":
        return wordnet.NOUN
    elif tag == "V":
        return wordnet.VERB
    elif tag == "R":
        return wordnet.ADV
    # Return empty if the given word is not one of the above tags
    return ""


def lemmatize(word: str) -> str:
    """
    Return lemmatized form of given word if it is a noun, verb, adjective or adverb.

    Reference: https://github.com/Proto007/Topic-Modeling-Book-Descriptions/blob/404f4700b73fbcedec55136a85f68ebcc21580f0/backend/services/model/dataPreprocess.py#L72

    @params:
        word (string): query word
    @return
        lemmatized_word (string): lemmatized word if it is one of the valid pos tags/ returns the word itself if it is empty or not a valid pos tag
    """
    # Return the word itself if its empty
    if not word:
        return word
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Get the pos tag for the word
    pos_tag = get_pos_tag(word)
    # If the pos tag is not valid, return the word itself
    if not pos_tag:
        return word
    # Return the lemmatized form of the word
    return lemmatizer.lemmatize(word, pos_tag)


def preprocess(post: str):
    """
    Apply tokenizing, filter stopwords, lemmatizing, case-folding to the given post

    @params:
        post (string): an unprocessed post
    @return:
        preprocessed_post (string): apply preprocessing to the parameter post and return preprocessed post
    """
    # tokenize the words in given post
    processed = word_tokenize(post)
    # filter stopwords in the post
    processed = [
        word.strip() for word in processed if word not in stopwords.words("english")
    ]
    # casefold all words in post to lower-case
    processed = [word.lower() for word in processed if len(word) > 1]
    # lemmatize the words in post
    processed = [lemmatize(word) for word in processed]
    # join all the processed words into a string separated by spaces and return
    return " ".join(processed)


class TrainModelViewset(viewsets.ModelViewSet):
    queryset = TrainModel.objects.all()
    serializer_class = TrainModelSerializer

    def create(self, request):
        try:
            df = pd.read_csv(request.data["dataset"])
        except Exception:
            return Response(
                {"message": "pandas failed to read file"},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )
        preprocessed = request.data.get("preprocessed", False)
        TrainModel.objects.all().delete()
        if os.path.isdir(f"{BASE_DIR}/dataset"):
            shutil.rmtree(f"{BASE_DIR}/dataset")
        if not preprocessed:
            df["posts"].apply(lambda p: preprocess(p))
        train_x, test_x, train_y, test_y = model_selection.train_test_split(
            df["posts"], df["type"], test_size=0.2
        )
        tfidf = TfidfVectorizer()
        tfidf.fit_transform(df["posts"])
        train_x_tfidf = tfidf.transform(train_x)
        classifier = CalibratedClassifierCV(LinearSVC())
        classifier.fit(train_x_tfidf, train_y)
        text_classifier = Pipeline(
            [("tfidf", TfidfVectorizer()), ("classifier", CalibratedClassifierCV())]
        )
        text_classifier.fit(train_x, train_y)
        pickle.dump(text_classifier, open("new_svm_model.sav", "wb"))
        predictions = text_classifier.predict(test_x)
        accuracy = round(accuracy_score(test_y, predictions), 2)
        TrainModel.objects.create(
            dataset=request.data["dataset"],
            preprocessed=bool(preprocessed),
            accuracy=accuracy,
        )
        return Response(status=status.HTTP_200_OK)


class ClassifyViewset(viewsets.ModelViewSet):
    queryset = Classify.objects.all()
    serializer_class = ClassifySerializer

    def create(self, request):
        serializer = ClassifySerializer
        use_default = request.data.get("use_default", False)
        if not use_default:
            if not os.path.isfile("new_svm_model.sav"):
                return Response(
                    {"message": "Model not found. Try using default."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            svm_clf = pickle.load(open("new_svm_model.sav", "rb"))
        else:
            svm_clf = pickle.load(open("default_svm_model.sav", "rb"))
        query = preprocess(request.data["query_text"])
        predictions = svm_clf.predict_proba([query])
        prob_dict = {svm_clf.classes_[n]: predictions[0][n] for n in range(16)}
        top_5 = " ".join(
            sorted(list(prob_dict.keys()), key=lambda x: prob_dict[x], reverse=True)[:5]
        )
        new_classification = Classify.objects.create(
            query_text=request.data["query_text"],
            use_default=bool(use_default),
            prediction=top_5,
        )
        new_classification.save()
        return Response(serializer(new_classification).data, status=status.HTTP_200_OK)
