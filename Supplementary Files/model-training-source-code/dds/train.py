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
# from keras.callbacks import ModelCheckpoint, EarlyStopping


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
    # df = pd.read_csv('/Users/orphic/Downloads/datasets/tissue/train/run_' + str(stats['gen']) + '.csv')
    # df1 = pd.read_csv('/Users/orphic/Downloads/datasets/tissue/val/run_' + str(stats['gen']) + '.csv')
    # df['label'] = df['label'].astype(str)
    # df1['label'] = df1['label'].astype(str)
    # train_generator = train_datagen.flow_from_dataframe(df,
    #                                     directory="/Users/orphic/Downloads/maindatasets/tmp/tissuemnist",
    #                                     x_col="image",
    #                                     y_col="label",
    #                                     target_size=(28, 28),
    #                                     batch_size=32,
    #                                     class_mode='categorical', color_mode = "grayscale"
    #                                     )
    # test_generator = train_datagen.flow_from_dataframe(df1,
    #                                     directory="/Users/orphic/Downloads/maindatasets/tmp/tissuemnist",
    #                                     x_col="image",
    #                                     y_col="label",
    #                                     target_size=(28, 28),
    #                                     batch_size=32,
    #                                       class_mode='categorical',color_mode = "grayscale"
                #   40 4 same 1 tanh 52 3 same 1 32 SGD(lr=float(0.01),momentum=float(0.82)) 352 mnist
# 64 3 valid 1 tanh 40 5 same 2 416 Adam(learning_rate=float(0.0001)) 512	0.5180000066757202 cifar 10_fial# 40 3 valid 2 tanh
# 16 4 same 3 256 Adam(learning_rate=float(0.001)) 416
train_generator = train_datagen.flow_from_directory('/Users/orphic/Downloads/maindatasets/TinyImageNet/train',class_mode='categorical', batch_size=32, target_size=(64, 64), color_mode = "rgb")  # since we use binary_crossentropy loss, we need binary labels
test_generator = train_datagen.flow_from_directory('/Users/orphic/Downloads/maindatasets/TinyImageNet/val',class_mode='categorical', batch_size=32, target_size=(64, 64), color_mode = "rgb")  # since we use binary_crossentropy loss, we need binary labels
model=Sequential()
model.add(Conv2D(filters=48, kernel_size=(5,5), activation= "tanh", input_shape=(64, 64, 3), padding='same'))
model.add(MaxPool2D(strides=1))
model.add(Conv2D(filters=52, kernel_size=(3,3), activation="tanh", padding='same'))
model.add(MaxPool2D(strides=3))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dense(320, activation='tanh'))
model.add(Dense(100, activation='softmax')) #no of output classes and last layer, hence softmax
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = Adamax(lr=float(0.001)), metrics=["accuracy"])
hist = model.fit_generator(generator= train_generator, epochs= 100, validation_data= test_generator)
# print("gen" + str(stats['gen']))                

# maximum_epochs = 1000
# early_stop_epochs = 10
# learning_rate_epochs = 5
f1 = hist.history['val_accuracy']
f2 = hist.history['val_loss']
f3 = hist.history['accuracy']
f4 = hist.history['loss']
f = open("cifar100_optuna.txt",'a')
f.writelines(str(f1) + '\t')
f.writelines(str(f2) + '\t')
f.writelines(str(f3) + '\t')
f.writelines(str(f4) + '\t')
f.writelines('\n')
f.close()

model_json = model.to_json()
with open("imagenet_optuna.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("imagenet_optuna.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('imagenet_optuna.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("imagenet_optuna.h5")
print("Loaded model from disk")