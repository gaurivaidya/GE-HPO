import os
import random
from shutil import copyfile

dataset_dir = '/home/gauri/caltech-101/101_ObjectCategories'

train_dir = '/home/gauri/caltech-101/train'
val_dir = '/home/gauri/caltech-101/val'
test_dir = '/home/gauri/caltech-101/test'

classes = os.listdir(dataset_dir)
for cls in classes:
    img_dir = os.path.join(dataset_dir, cls)
    imgs = os.listdir(img_dir)
    random.shuffle(imgs)
    split1 = int(len(imgs)*0.6) #60% train
    split2 = int(len(imgs)*0.8) #20% val, 20% test
    
    for img in imgs[:split1]:
        if not os.path.exists(os.path.join(train_dir, cls)):
            os.makedirs(os.path.join(train_dir, cls))
        copyfile(os.path.join(img_dir, img), os.path.join(train_dir, cls, img))
        
    for img in imgs[split1:split2]:
        if not os.path.exists(os.path.join(val_dir, cls)):
            os.makedirs(os.path.join(val_dir, cls))
        copyfile(os.path.join(img_dir, img), os.path.join(val_dir, cls, img))
        
    for img in imgs[split2:]:
        if not os.path.exists(os.path.join(test_dir, cls)):
            os.makedirs(os.path.join(test_dir, cls))
        copyfile(os.path.join(img_dir, img), os.path.join(test_dir, cls, img))
