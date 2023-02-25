from django.contrib import admin

from .models import *

# Register models to admin debug site
admin.register(Classify)
admin.register(TrainModel)
