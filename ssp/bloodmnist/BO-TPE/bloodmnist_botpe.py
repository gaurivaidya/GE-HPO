from inspect import Parameter
from random import shuffle
import optuna
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import numpy as np 
import itertools
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from random import shuffle
import numpy as np
import re
from csv import writer
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import f1_score
from torchvision.models import resnet50
from torchvision import datasets, models, transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import warnings
from ptflops import get_model_complexity_info
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator

warnings.filterwarnings("ignore")

def objective(trial): 
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128]) 
    optimiser = trial.suggest_categorical('optimiser', ['adam', 'sgd', 'rmsprop', 'adamax', 'adagrad', 'adadelta'])
    lr = trial.suggest_uniform('lr', 0, 1)
    momentum = trial.suggest_uniform('momentum', 0.6, 0.9)

                          
    dict_params = {'batch_size': batch_size,
                   'optimiser': optimiser,
                   'lr': lr,
                   'momentum': momentum}
                                            
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_flag = 'bloodmnist'
    download = True

    NUM_EPOCHS = 5
    BATCH_SIZE = dict_params['batch_size']
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


    # class Net(nn.Module):
    def set_parameter_requires_grad(model, feature_extracting):
                    if feature_extracting:
                        for param in model.parameters():
                            param.requires_grad = False
    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
                    # Initialize these variables which will be set in this if statement. Each of these
                    #   variables is model specific.
        model_ft = None
        input_size = 0 
        model_ft = models.resnet50()
        # print("this one2")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model_ft
    model = initialize_model("resnet", 8, True, use_pretrained=True)
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        # print("this one4")
    if str(dict_params['optimiser']) == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=float(dict_params['lr']), momentum=float(dict_params['momentum']))
    elif str(dict_params['optimiser']) == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=float(dict_params['lr']))
    elif str(dict_params['optimiser']) == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=float(dict_params['lr']), momentum=float(dict_params['momentum']))
    elif str(dict_params['optimiser']) == 'adamax':
            optimizer = optim.Adamax(model.parameters(), lr=float(dict_params['lr']))
    elif str(dict_params['optimiser']) == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=float(dict_params['lr']))
    elif str(dict_params['optimiser']) == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=float(dict_params['lr']))

    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        
        model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
    def test(split):
        model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        data_loader = train_loader_at_eval if split == 'train' else test_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            
            evaluator = Evaluator(data_flag, split)
            metrics = evaluator.evaluate(y_score)
        
            print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
            return metrics

            
    print('==> Evaluating ...')
    a1 = test('train')
    b = test('test')
    end = time.time()


    final_fitness =  float(('test', *b)[2])
    return final_fitness

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=10)
optuna.logging.set_verbosity(optuna.logging.WARNING)
    
# save results
df_results = study.trials_dataframe()
df_results.to_csv('medmnist_optuna.csv')