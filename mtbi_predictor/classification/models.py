from django.db import models


class Classify(models.Model):
    """
    `Classify` model contains an input `query_text` field and `prediction` readonly field
    """

    GROUPS = (
        ("INTJ", "INTJ"),
        ("INTP", "INTP"),
        ("ENTJ", "ENTJ"),
        ("ENTP", "ENTP"),
        ("INFJ", "INFJ"),
        ("INFP", "INFP"),
        ("ENFJ", "ENFJ"),
        ("ENFP", "ENFP"),
        ("ISTJ", "ISTJ"),
        ("ESTJ", "ESTJ"),
        ("ESFJ", "ESFJ"),
        ("ISTP", "ISTP"),
        ("ISFP", "ISFP"),
        ("ESTP", "ESTP"),
        ("ESFP", "ESFP"),
    )
    query_text = models.TextField(max_length=5000)
    prediction = models.CharField(max_length=4, choices=GROUPS)
