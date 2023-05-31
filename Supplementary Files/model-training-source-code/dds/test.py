import optuna
import matplotlib.pyplot as plt
# from inspect import Parameter
# from random import shuffle
# from algorithm.parameters import params
# from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
import re
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import f1_score
import tensorflow.keras
import tensorflow as tf
from sklearn import metrics
from keras.models import Sequential
from stats.stats import stats
from tensorflow.keras.models import Sequential, model_from_json


from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense, AveragePooling2D, Conv2D, MaxPool2D 
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

#####imports#######

import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from keras.models import load_model

# from keras.callbacks import ModelCheckpoint, EarlyStopping
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory('/Users/orphic/Downloads/maindatasets/TinyImageNet/test',class_mode='categorical', batch_size=32, target_size=(64, 64), color_mode = "rgb")  # since we use binary_crossentropy loss, we need binary labels

json_file = open('HyperGE_main/imagenet_hyperge.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("HyperGE_main/imagenet_hyperge.h5")

loaded_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=float(0.001)), metrics=["accuracy"])
loss, acc = loaded_model.evaluate_generator(test_generator)

print(loss)
print(acc)


