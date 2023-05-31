import optuna
import matplotlib.pyplot as plt
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


# define hyperparameter search space 
def objective(trial):        
    filters1 = trial.suggest_categorical('filters1', [16, 24, 32, 40, 48, 52, 64])
    filters2 = trial.suggest_categorical('filters2', [16, 24, 32, 40, 48, 52, 64])
    kernel_size1 = trial.suggest_categorical('kernel_size1', [3, 4, 5])
    strides1 = trial.suggest_categorical('strides1', [1, 2, 3, 4])
    padding1 = trial.suggest_categorical('padding1', ["same", "valid"])
    dense1 = trial.suggest_categorical('dense1', [32, 64 ,96 ,128 ,160 ,192 ,224 , 256 ,288 ,320 ,352 ,384 ,416 ,448 ,480 ,512])
    kernel_size2 = trial.suggest_categorical('kernel_size2', [3, 4, 5])
    strides2 = trial.suggest_categorical('strides2', [1, 2, 3, 4])
    padding2 = trial.suggest_categorical('padding2', ["same", "valid"])
    dense2 = trial.suggest_categorical('dense2', [32, 64 ,96 ,128 ,160 ,192 ,224 , 256 ,288 ,320 ,352 ,384 ,416 ,448 ,480 ,512])
    optimiser = trial.suggest_categorical('optimiser2', [Adam, SGD, RMSprop, Adamax])
    lr = trial.suggest_categorical('lr', [0.00001 , 0.0001 , 0.001 , 0.01 , 0.1])
    actfunc = trial.suggest_categorical('actfunc', ['elu', 'relu', 'selu', 'sigmoid', 'tanh'])

                          
    dict_params = {'filters1':filters1,
                    'filters2':filters2,
                    'kernel_size1':kernel_size1,
                    'strides1':strides1,
                    'padding1':padding1,
                    'dense1':dense1,
                    'kernel_size2':kernel_size2,
                    'strides2':strides2,
                    'padding2':padding2,
                    'dense2':dense2,
                    'optimiser':optimiser,
                    'lr':lr,
                    'actfunc':actfunc}
                                            
    # start of cnn coding   
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    # use subsets of data instead of entire dataset
    
    if (trial.number >=20 and trial.number < 40) or (trial.number >=40 and  trial.number < 60) or (trial.number >=60 and  trial.number < 80):
        y=1
    else:
        y = trial.number  
    
    train_generator = train_datagen.flow_from_directory('/Users/orphic/Downloads/datasets/imagenet/run_' + str(y) + '/train',class_mode='categorical', batch_size=32, target_size=(32, 32), color_mode = "rgb")  # since we use binary_crossentropy loss, we need binary labels
    test_generator = train_datagen.flow_from_directory('/Users/orphic/Downloads/datasets/imagenet2/run_' + str(y) +  '/val',class_mode='categorical', batch_size=32, target_size=(32, 32), color_mode = "rgb")  # since we use binary_crossentropy loss, we need binary labels
    y = y + 1
    model=Sequential()
    model.add(Conv2D(filters=dict_params['filters1'], kernel_size=(dict_params['kernel_size1'], dict_params['kernel_size1']), activation= dict_params['actfunc'], input_shape=(32, 32, 3)))
    model.add(MaxPool2D(strides=dict_params['strides1']))
    model.add(Conv2D(filters=dict_params['filters2'], kernel_size=(dict_params['kernel_size2'], dict_params['kernel_size2']), activation=dict_params['actfunc']))
    model.add(MaxPool2D(strides=dict_params['strides2']))
    model.add(Flatten())
    model.add(Dense(dict_params['dense1'], activation=dict_params['actfunc']))
    model.add(Dense(dict_params['dense2'], activation=dict_params['actfunc']))
    model.add(Dense(100, activation='softmax')) #no of output classes and last layer, hence softmax
    model.summary()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = dict_params['optimiser'](lr=dict_params['lr']), metrics=["accuracy"])
    hist = model.fit_generator(generator= train_generator, epochs= 5, validation_data= test_generator)
    optimizer_direction = 'maximise'
    f1 = hist.history['val_accuracy']
    final_fitness = float(f1[-1])

    number_of_random_points = 5  # random searches to start opt process
    return final_fitness

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=80, n_jobs=10)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# save results
df_results = study.trials_dataframe()
df_results.to_csv('imaganet_optuna.csv')