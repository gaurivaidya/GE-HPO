from inspect import Parameter
from random import shuffle
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
import re
from csv import writer
from flopth import flopth
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import f1_score
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

warnings.filterwarnings("ignore")





class hyperge_2(base_ff):
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
            # Top level data directory. Here we assume the format of the directory conforms
            #   to the ImageFolder structure
            data_dir = "/home/Gauri/cifar100_main"

            # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
            model_name = "resnet"

            # Number of classes in the dataset
            num_classes = 100

            # Batch size for training (change depending on how much memory you have)
            batch_size = int(a[3])

            # Number of epochs to train for
            num_epochs = 5

            # Flag for feature extracting. When False, we finetune the whole model,
            #   when True we only update the reshaped layer params
            feature_extract = True
            def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
                since = time.time()

                val_acc_history = []

                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = 0.0

                for epoch in range(num_epochs):
                    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    print('-' * 10)

                    # Each epoch has a training and validation phase
                    for phase in ['train', 'val']:
                        if phase == 'train':
                            model.train()  # Set model to training mode
                        else:
                            model.eval()   # Set model to evaluate mode

                        running_loss = 0.0
                        running_corrects = 0

                        # Iterate over data.
                        for inputs, labels in dataloaders[phase]:
                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            with torch.set_grad_enabled(phase == 'train'):
                                # Get model outputs and calculate loss
                                # Special case for inception because in training it has an auxiliary output. In train
                                #   mode we calculate the loss by summing the final output and the auxiliary output
                                #   but in testing we only consider the final output.
                                if is_inception and phase == 'train':
                                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                    outputs, aux_outputs = model(inputs)
                                    loss1 = criterion(outputs, labels)
                                    loss2 = criterion(aux_outputs, labels)
                                    loss = loss1 + 0.4*loss2
                                else:
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)

                                _, preds = torch.max(outputs, 1)

                                # backward + optimize only if in training phase
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)

                        epoch_loss = running_loss / len(dataloaders[phase].dataset)
                        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                        # deep copy the model
                        if phase == 'val' and epoch_acc > best_acc:
                            best_acc = epoch_acc
                            best_model_wts = copy.deepcopy(model.state_dict())
                        if phase == 'val':
                            val_acc_history.append(epoch_acc)

                    print()

                train_model.time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(train_model.time_elapsed // 60, train_model.time_elapsed % 60))
                print('Best val Acc: {:4f}'.format(best_acc))

                # load best model weights
                model.load_state_dict(best_model_wts)
                return model, val_acc_history
            def set_parameter_requires_grad(model, feature_extracting):
                if feature_extracting:
                    for param in model.parameters():
                        param.requires_grad = False
            def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
                # Initialize these variables which will be set in this if statement. Each of these
                #   variables is model specific.
                model_ft = None
                input_size = 0

                if model_name == "resnet":
                    """ Resnet18
                    """ 
                    model_ft = models.resnet50(pretrained=use_pretrained)
                    set_parameter_requires_grad(model_ft, feature_extract)
                    num_ftrs = model_ft.fc.in_features
                    model_ft.fc = nn.Linear(num_ftrs, num_classes)
                    input_size = 224
                return model_ft, input_size
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # Data augmentation and normalization for training
            # Just normalization for validation
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
            # Create training and validation dataloaders
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

            # Detect if we have a GPU available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Send the model to GPU
            model_ft = model_ft.to(device)

            # Gather the parameters to be optimized/updated in this run. If we are
            #  finetuning we will be updating all parameters. However, if we are
            #  doing feature extract method, we will only update the parameters
            #  that we have just initialized, i.e. the parameters with requires_grad
            #  is True.
            params_to_update = model_ft.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t",name)
            else:
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\t",name)

            # Observe that all parameters are being optimized
            if str(a[0]) == 'sgd':
                optimizer_ft = optim.SGD(params_to_update, lr=float(a[1]), momentum=float(a[2]))
            elif str(a[0]) == 'adam':
                optimizer_ft = optim.Adam(params_to_update, lr=float(a[1]))
            elif str(a[0]) == 'rmsprop':
                optimizer_ft = optim.RMSprop(params_to_update, lr=float(a[1]), momentum=float(a[2]))
            elif str(a[0]) == 'adamax':
                optimizer_ft = optim.Adamax(params_to_update, lr=float(a[1]))
            elif str(a[0]) == 'adadelta':
                optimizer_ft = optim.Adadelta(params_to_update, lr=float(a[1]))
            elif str(a[0]) == 'adagrad':
                optimizer_ft = optim.Adagrad(params_to_update, lr=float(a[1]))

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, 
                                        num_epochs=num_epochs)
            
            from pthflops import count_ops
            macs, params = get_model_complexity_info(model_ft, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)


 
            
                

            final_fitness = float(max(hist).item())
            # List that we want to add as a new row
            List = [a[0], a[1], a[2], a[3], final_fitness, macs, params,train_model.time_elapsed]

            with open('cifar100_hyperge.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(List)
                # Close the file object
                f_object.close()

            return final_fitness   


