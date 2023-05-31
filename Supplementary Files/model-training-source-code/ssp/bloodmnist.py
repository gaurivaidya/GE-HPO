#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('pip install medmnist')
from torchvision import datasets, models, transforms
 

# In[1]:


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


# In[2]:


print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")


# # We first work on a 2D dataset

# In[3]:


data_flag = 'bloodmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 100
BATCH_SIZE = 64
lr = 0.23

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# ## First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form.

# In[4]:


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


# In[5]:


print(train_dataset)
print("===================")
print(test_dataset)


# In[ ]:





# In[6]:


# visualization

# train_dataset.montage(length=1)


# In[8]:


# montage

# train_dataset.montage(length=20)


# ## Then, we define a simple model for illustration, object function and optimizer that we use to classify.

# In[9]:


# define a simple CNN model

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

                if model_name == "resnet":
                    """ Resnet50
                    """ 
                    model_ft = models.resnet50(pretrained=use_pretrained)
                    set_parameter_requires_grad(model_ft, feature_extract)
                    num_ftrs = model_ft.fc.in_features
                    model_ft.fc = nn.Linear(num_ftrs, num_classes)
                    input_size = 224
                return model_ft
model = initialize_model('resnet', 8, True, use_pretrained=True)
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.Adadelta(model.parameters(), lr=lr)


# ## Next, we can start to train and evaluate!

# In[10]:


# train

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


# In[11]:


# evaluation

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
        print(float((split, *metrics)[2]))

        
print('==> Evaluating ...')
test('train')
test('test')


# In[ ]:





# # # We then check a 3D dataset

# # In[12]:


# data_flag = 'bloodmnist'
# download = True

# info = INFO[data_flag]
# DataClass = getattr(medmnist, info['python_class'])

# # load the data
# train_dataset = DataClass(split='train',  download=download)

# # encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# # In[13]:


# x, y = train_dataset[0]

# print(x.shape, y.shape)


# # In[14]:


# for x, y in train_loader:
#     print(x.shape, y.shape)
#     break


# # In[15]:


# frames = train_dataset.montage(length=1, save_folder="tmp/")
# frames[10]


# # In[16]:


# frames = train_dataset.montage(length=20, save_folder="tmp/")

# frames[10]


# # ## Go and check the generated [gif](tmp/organmnist3d_train_montage.gif) ;)

# # # Check [EXTERNAL] [`MedMNIST/experiments`](https://github.com/MedMNIST/experiments)
# # 
# # Training and evaluation scripts to reproduce both 2D and 3D experiments in our paper, including PyTorch, auto-sklearn, AutoKeras and Google AutoML Vision together with their weights ;)

# # In[ ]:




