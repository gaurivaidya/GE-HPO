from inspect import Parameter
from random import shuffle
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
import re
from csv import writer
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import f1_score
# import tensorflow as tf
from torchvision.models import resnet50, ResNet50_Weights
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
from stats.stats import stats
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
warnings.filterwarnings("ignore")







class hyperge1(base_ff):
    """Fitness function for tuning the optimal hyperparameter for a five layer LeNet model."""
    maximise = True
    global hist

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        maximise = True

    def evaluate(self, ind, **kwargs):
        start = time.time()
        string_phenotype = str(ind.phenotype)
        clean = re.sub('[^A-Za-z0-9]+\.[0-9]+', ' ', string_phenotype)
        a = clean.split() # array of all hyperparameters
        data_flag = 'bloodmnist'
        download = True

        NUM_EPOCHS = 5
        BATCH_SIZE = int(a[3])

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
        if str(a[0]) == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=float(a[1]), momentum=float(a[2]))
        elif str(a[0]) == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=float(a[1]))
        elif str(a[0]) == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=float(a[1]), momentum=float(a[2]))
        elif str(a[0]) == 'Adamax':
                optimizer = optim.Adamax(model.parameters(), lr=float(a[1]))
        elif str(a[0]) == 'Adadelta':
                optimizer = optim.Adadelta(model.parameters(), lr=float(a[1]))
        elif str(a[0]) == 'Adagrad':
                optimizer = optim.Adagrad(model.parameters(), lr=float(a[1]))

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
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                    print_per_layer_stat=True, verbose=True)
        end = time.time()

 
        final_fitness =  float(('test', *b)[2])

        # # List that we want to add as a new row
        List = [a[0], a[1], a[2], a[3], final_fitness, macs, params,(end-start)/60]

        with open('medmnist_hyperge.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            # Close the file object
            f_object.close()
        return final_fitness