from inspect import Parameter
from random import shuffle
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
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
from keras import backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense, AveragePooling2D, Conv2D, MaxPool2D 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax






class lenet(base_ff):
    """Fitness function for tuning the optimal hyperparameter for a five layer LeNet model."""
    maximise = True
    global hist

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        maximise = True

    def evaluate(self, ind, **kwargs):
            string_phenotype = str(ind.phenotype)
            clean = re.sub('[^A-Za-z0-9]+\.[0-9]+', ' ', string_phenotype)
            a = clean.split() # array of all hyperparameters
            train_datagen = ImageDataGenerator()
            test_datagen = ImageDataGenerator() 
            # following command specifies the dataset directory. replace with relative dataset path
            train_generator = train_datagen.flow_from_directory('/Users/orphic/Downloads/datasets/mnist/run_' + str(stats['gen']) + '/train',class_mode='categorical', batch_size=32, target_size=(28, 28), color_mode = "grayscale")  # since we use binary_crossentropy loss, we need binary labels
            test_generator = train_datagen.flow_from_directory('/Users/orphic/Downloads/datasets/mnist/run_' + str(stats['gen'])+'/val',class_mode='categorical', batch_size=32, target_size=(28, 28), color_mode = "grayscale")  # since we use binary_crossentropy loss, we need binary labels
            # define model structure
            model=Sequential()
            model.add(Conv2D(filters=int(a[0]), kernel_size=(int(a[1]),int(a[1])), padding=a[2], activation=a[4], input_shape=(28, 28, 1)))
            model.add(MaxPool2D(strides=int(a[3])))
            model.add(Conv2D(filters=int(a[5]), kernel_size=(int(a[6]),int(a[6])), padding=a[7], activation=a[4]))
            model.add(MaxPool2D(strides=int(a[8])))
            model.add(Flatten())
            model.add(Dense(int(a[9]), activation=a[4]))
            model.add(Dense(int(a[11]), activation=a[4]))
            model.add(Dense(100, activation='softmax')) #no of output classes and last layer, hence softmax
            opt = eval(a[10])   
            model.summary()
            model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=["accuracy"])
            hist = model.fit_generator(generator= train_generator, epochs= 5, validation_data= test_generator)
            f1 = hist.history['val_accuracy']
            final_fitness = float(f1[-1])

            return final_fitness   


