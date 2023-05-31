import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.core.display import display
# from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy# regular imports
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger

import pickle

# pytorch related imports
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

# lightning related imports
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './data'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
                                                  ])
        
        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
                                                  ])

        
    def prepare_data(self):
        # download 
        torchvision.datasets.CIFAR100(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = torchvision.datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_train)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=64, num_workers=0)
    


early_stop_callback = EarlyStopping(
   monitor='val_loss',
   patience=48,
   verbose=False,
   mode='min'
)


MODEL_CKPT_PATH = './resnet50_model/'
MODEL_CKPT = 'resnet50-{epoch:02d}-{val_loss:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_CKPT_PATH ,
    filename=MODEL_CKPT ,
    save_top_k=3,
    mode='min')

def create_model():
    model = torchvision.models.resnet50(pretrained=False, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

# def create_model():
#     model = torchvision.models.resnet50(pretrained=True)
#     num_ftrs = model.fc.in_features
#     num_classes = 100
#     model.fc = nn.Linear(num_ftrs, num_classes)
#     return model

class LitResnet50(pl.LightningModule):
    def __init__(self, batch_size, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = create_model()
  
    # will be used during inference
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=0.028,)
        steps_per_epoch = 45000 // self.batch_size
        #scheduler_dict = { "scheduler": OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch,),
         #                  "interval": "step",}
        #return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return optimizer
    

# Init our data pipeline
dm = CIFAR100DataModule(batch_size=64)
     
    
# To access the x_dataloader we need to call prepare_data and setup.
dm.prepare_data()

dm.setup()

testloader = dm.test_dataloader()
trainloader = dm.train_dataloader()

#function to read files present in the Python version of the dataset
# def unpickle(file):
#     with open(file, 'rb') as fo:
#         myDict = pickle.load(fo, encoding='latin1')
#     return myDict

# metaData = unpickle('/content/data/cifar-100-python/meta')
# label_names = metaData['fine_label_names']
# print(len(label_names))

# function to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random test images to see if things are getting displayed correctly
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# labels_list = labels.tolist()
# for i in range(len(labels_list)):
#     print(labels_list[i], label_names[labels_list[i]] )

# model_save_path = '/content/gdrive/MyDrive/EMLO_S9_CIFAR100_Sagemaker/Resnet34_pl_cifar100.pt'
# model_save_path_cpu = '/content/gdrive/MyDrive/EMLO_S9_CIFAR100_Sagemaker/Resnet34_pl_cifar100_cpu.pt'


model = LitResnet50(dm.batch_size)

# if os.path.exists(model_save_path):
#     model.load_state_dict(torch.load(model_save_path))
#     print('Model loaded')
# else:
#     print('Starting with fresh model')

    # Initialize a trainer
trainer = pl.Trainer(max_epochs=100,
                     progress_bar_refresh_rate=20, 
                     gpus=1, 
                     logger=CSVLogger(save_dir="logs/"),
                     enable_model_summary=True,
                     callbacks=[early_stop_callback,
                                ModelSummary(max_depth=-1)],
                     checkpoint_callback=checkpoint_callback)
# Train the model again
trainer.fit(model, dm)

# Evaluate the model on the held out test set
trainer.test(model, dm)


# class LitResnet(LightningModule):
#     def __init__(self, lr=0.05):
#         super().__init__()

#         self.save_hyperparameters()
#         self.model = create_model()

#     def forward(self, x):
#         out = self.model(x)
#         return F.log_softmax(out, dim=1)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         self.log("train_loss", loss)
#         return loss

#     def evaluate(self, batch, stage=None):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy(preds, y)

#         if stage:
#             self.log(f"{stage}_loss", loss, prog_bar=True)
#             self.log(f"{stage}_acc", acc, prog_bar=True)

#     def validation_step(self, batch, batch_idx):
#         self.evaluate(batch, "val")

#     def test_step(self, batch, batch_idx):
#         self.evaluate(batch, "test")

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adagrad(
#             self.parameters(),
#             lr=0.04,
#             weight_decay=5e-4,
#         )
#         steps_per_epoch = 45000 // 64
#         # scheduler_dict = {
#         #     "scheduler": OneCycleLR(
#         #         optimizer,
#         #         0.1,
#         #         epochs=self.trainer.max_epochs,
#         #         steps_per_epoch=steps_per_epoch,
#         #     ),
#         #     "interval": "step",
#         # }
#         # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
#         return {"optimizer": optimizer}
    



# model = LitResnet(lr=0.05)

# dm = CIFAR100DataModule(batch_size=64)
     
    
# # To access the x_dataloader we need to call prepare_data and setup.
# dm.prepare_data()

# dm.setup()

# testloader = dm.test_dataloader()
# trainloader = dm.train_dataloader()

# trainer = Trainer(
#     max_epochs=500,
#     accelerator="auto",
#     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
#     logger=CSVLogger(save_dir="logs/"),
#     callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
# )

# trainer.fit(model, trainloader)
# trainer.test(model, datamodule=testloader)


metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sn.relplot(data=metrics, kind="line")
